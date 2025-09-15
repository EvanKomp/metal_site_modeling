# metalsitenn/nn/backbone.py
'''
* Author: Evan Komp
* Created: 8/13/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT


This module provides the main EquiformerWEdgesBackbone model that combines all the enhanced
components for molecular/protein modeling with rich atom/bond features and topology information.
'''

import contextlib
import logging
import typing
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.models.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from fairchem.core.models.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer
from fairchem.core.models.equiformer_v2.layer_norm import get_normalization_layer
from fairchem.core.models.equiformer_v2.module_list import ModuleListInfo
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Grid,
    SO3_Rotation,
)
from fairchem.core.models.equiformer_v2.weight_initialization import eqv2_init_weights

from .embedding import NodeEmbedder, SO3ScalarEmbedder, EdgeDegreeEmbedding
from .topology import compute_positional_topology_gradients, SO3_L1_LinearMixing
from .attention import TransBlockV2WithEdges
from .film import SO3EquivariantFiLM

with contextlib.suppress(ImportError):
    pass

if typing.TYPE_CHECKING:
    from metalsitenn.dataloading import BatchProteinData

# Statistics of IS2RE 100K (same as original)
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773


class EquiformerWEdgesBackbone(nn.Module):
    """
    Enhanced Equiformer backbone with molecular feature support and edge information processing.
    
    This is the core neural network architecture for molecular/protein modeling that combines SO(3)-equivariant
    graph neural networks with rich molecular features, topology information, and optional time-dependent
    modulation via FiLM (Feature-wise Linear Modulation).
    
    The model is designed to accept BatchProteinData with:
    - Precomputed molecular graphs with rich atom/bond features
    - Topology information for L1 gradient computation
    - Enhanced edge information processing in attention layers
    - Optional time-dependent features for diffusion-like models
    
    Architecture Overview:
    1. Node embedding: Maps molecular atom features to SO(3) equivariant representations
    2. Edge embedding: Incorporates bond features and distance information
    3. Transformer blocks: SO(3)-equivariant attention with enhanced edge processing
    4. Topology mixing: Optional gradient-based features for molecular constraints
    5. FiLM modulation: Optional time-dependent feature modulation
    
    Args:
        # Core Architecture Parameters
        num_layers (int, default=12):
            Number of transformer layers in the backbone.
            	Used by: _init_enhanced_transformer_blocks() to create TransBlockV2WithEdges layers
            	Higher values increase model capacity but also computational cost
        
        sphere_channels (int, default=128):
            Number of spherical harmonic channels per resolution.
            	Used by: All SO3 operations, embeddings, and transformer blocks
            	Determines the feature dimensionality in spherical harmonic space
            	Total channels = sphere_channels * num_resolutions
        
        attn_hidden_channels (int, default=128):
            Hidden channels for SO(2) graph attention computations.
            	Used by: TransBlockV2WithEdges attention modules
            	Controls attention computation capacity independent of main feature channels
        
        num_heads (int, default=8):
            Number of attention heads in multi-head attention.
            	Used by: TransBlockV2WithEdges attention modules
            	Must divide evenly into attn_alpha_channels and attn_value_channels
        
        attn_alpha_channels (int, default=32):
            Channels for alpha (query/key) vectors in attention heads.
            	Used by: SO2EquivariantGraphAttention in TransBlockV2WithEdges
            	Should be divisible by num_heads for even distribution
        
        attn_value_channels (int, default=16):
            Channels for value vectors in attention heads.
            	Used by: SO2EquivariantGraphAttention in TransBlockV2WithEdges
            	Should be divisible by num_heads for even distribution
        
        ffn_hidden_channels (int, default=512):
            Hidden channels in feedforward network layers.
            	Used by: FeedForwardNetwork in TransBlockV2WithEdges
            	Controls capacity of position-wise feedforward transformations
        
        norm_type (str, default="rms_norm_sh"):
            Type of normalization layer to use.
            	Used by: get_normalization_layer() for norm_1 and norm_final
            	Options: ['layer_norm', 'layer_norm_sh', 'rms_norm_sh']
            	'rms_norm_sh' is optimized for spherical harmonics
        
        # SO(3) Representation Parameters
        lmax_list (List[int], default=[6]):
            List of maximum degrees (l) for spherical harmonics per resolution.
            	Used by: SO3 components, embeddings, attention, all spherical operations
            	Determines angular resolution: l=0 (scalar), l=1 (vector), l=2 (quadrupole), etc.
            	Currently only single resolution supported (length must be 1)
        
        mmax_list (List[int], default=None):
            List of maximum orders (m) for spherical harmonics per resolution.
            	IGNORED: Automatically derived from lmax_list (mmax = lmax for each resolution)
            	Legacy parameter maintained for compatibility
        
        grid_resolution (int, default=None):
            Resolution for SO3 grid representations.
            	Used by: SO3_Grid modules for spherical harmonic grid computations
            	Higher values increase accuracy but computational cost
            	If None, uses default from SO3_Grid
        
        num_sphere_samples (int, default=128):
            Number of samples for sphere integration approximation.
            	Used by: Output blocks for integrating over spherical surfaces
            	Higher values improve integration accuracy
                IGNORED in this class but stored for use by heads maybe?
        
        # Edge and Distance Parameters
        edge_channels_list (List[int], default=None):
            List of channel sizes for invariant edge embeddings.
            	Used by: TransBlockV2WithEdges, EdgeDegreeEmbedding
            	Default: [num_distance_basis, sphere_channels, sphere_channels]
            	First element: input channels, subsequent: hidden channels
        
        use_m_share_rad (bool, default=False):
            Whether all m components within a type-L vector share radial function weights.
            	Used by: RadialFunction in SO3 operations
            	True: Reduces parameters, False: More expressive radial functions
        
        distance_function (str, default="gaussian_rbf"):
            Type of distance expansion function.
            	Used by: _init_distance_expansion() to create distance_expansion module
            	Options: ['gaussian', 'gaussian_rbf']
            	'gaussian_rbf': GaussianRadialBasisLayer (learnable)
            	'gaussian': GaussianSmearing (not learnable, fixed basis)
        
        num_distance_basis (int, default=128):
            Number of radial basis functions for distance expansion.
            	Used by: distance_expansion module, EdgeDegreeEmbedding, TransBlockV2WithEdges
            	Higher values allow finer distance resolution
        
        max_radius (float, default=5.0):
            Maximum radius for distance expansion.
            	Used by: distance_expansion module to set basis function range
            	Probably should be relatively large as it won't increase graph size.
        
        # Activation and Regularization Parameters
        attn_activation (str, default="scaled_silu"):
            Activation function for attention computations.
            	Used by: SO2EquivariantGraphAttention in TransBlockV2WithEdges
            	Controls non-linearity in attention score computation
        
        use_s2_act_attn (bool, default=False):
            Whether to use S2 activation after attention.
            	Used by: TransBlockV2WithEdges attention modules
            	Alternative to standard attention activation
        
        use_attn_renorm (bool, default=True):
            Whether to re-normalize attention weights.
            	Used by: SO2EquivariantGraphAttention in TransBlockV2WithEdges
            	Helps stabilize attention during training
        
        ffn_activation (str, default="scaled_silu"):
            Activation function for feedforward networks.
            	Used by: FeedForwardNetwork in TransBlockV2WithEdges
            	Controls non-linearity in position-wise transformations
        
        use_gate_act (bool, default=False):
            Whether to use gate activation instead of S2 activation.
            	Used by: TransBlockV2WithEdges for choosing activation type
            	Gate activation can be more stable but less expressive
        
        use_grid_mlp (bool, default=False):
            Whether to use grid-based MLPs in feedforward networks.
            	Used by: FeedForwardNetwork in TransBlockV2WithEdges
            	Projects to grids before applying MLPs for efficiency
        
        use_sep_s2_act (bool, default=True):
            Whether to use separable S2 activation when gate activation is disabled.
            	Used by: TransBlockV2WithEdges when use_gate_act=False
            	More efficient S2 activation implementation
        
        alpha_drop (float, default=0.0):
            Dropout rate for attention weights.
            	Used by: SO2EquivariantGraphAttention in TransBlockV2WithEdges
            	Helps prevent attention overfitting
        
        drop_path_rate (float, default=0.0):
            Drop path rate for stochastic depth regularization.
            	Used by: TransBlockV2WithEdges for residual connection dropping
            	Randomly drops entire residual branches during training
        
        proj_drop (float, default=0.0):
            Dropout rate for attention and feedforward outputs.
            	Used by: TransBlockV2WithEdges for output projections
            	Applied after attention and FFN computations
        
        weight_init (str, default="uniform"):
            Weight initialization method.
            	Used by: eqv2_init_weights() applied to all modules
            	Options: ['uniform', 'normal']
        
        # FiLM Time Modulation Parameters
        use_time (bool, default=False):
            Whether to expect time tensors and use FiLM modulation.
            	Used by: _init_film() and forward() for time-dependent feature modulation
            	Enables diffusion-like models with time-conditioned generation
        
        film_time_embedding_dim (int, default=128):
            Dimension for time embedding after nonlinear projection.
            	Used by: SO3EquivariantFiLM for time feature processing
            	Output dimension of time embedding MLP
        
        film_hidden_dim (int, default=256):
            Hidden dimension for FiLM MLP layers.
            	Used by: SO3EquivariantFiLM internal MLP computations
            	Controls capacity of time-to-modulation transformation
        
        film_mlp_layers (int, default=2):
            Number of layers in FiLM MLP.
            	Used by: SO3EquivariantFiLM for time embedding processing
            	Deeper MLPs can learn more complex time dependencies
        
        film_num_gaussians (int, default=512):
            Number of Gaussian basis functions for time encoding.
            	Used by: SO3EquivariantFiLM for time smearing/encoding
            	More Gaussians provide finer temporal resolution
        
        film_basis_function (str, default="gaussian_rbf"):
            Type of basis function for time embedding.
            	Used by: SO3EquivariantFiLM for time encoding
            	Options: ['gaussian_rbf', 'gaussian']
        
        # Enhanced Molecular Feature Parameters
        feature_vocab_sizes (Dict[str, int], default=None):
            Vocabulary sizes for categorical molecular features.
            	Used by: NodeEmbedder, EdgeDegreeEmbedding, TransBlockV2WithEdges
            	Maps feature names to vocabulary sizes for embedding layers
            	Example: {'element': 119, 'charge': 10, 'bond_order': 5}
        
        atom_features (List[str], default=['element', 'charge', 'nhyd', 'hyb']):
            List of atom feature names to use from BatchProteinData.
            	Used by: NodeEmbedder for atom embedding, _extract_feature_dict()
            	Must correspond to attributes in BatchProteinData
        
        bond_features (List[str], default=['bond_order', 'is_in_ring', 'is_aromatic']):
            List of bond feature names to use from BatchProteinData.
            	Used by: EdgeDegreeEmbedding, TransBlockV2WithEdges for edge embedding
            	Must correspond to edge attributes in BatchProteinData
        
        embedding_dim (int, default=32):
            Embedding dimension for categorical feature tokens.
            	Used by: NodeEmbedder, EdgeEmbedder, EdgeDegreeEmbedding, and SO2EquivariantGraphAttention (througj the node and edge embedders
            	Size of lookup table embeddings before linear projection
        
        edge_degree_projector_hidden_layers (int, default=2):
            Number of hidden layers in edge degree projector MLP.
            	Used by: EdgeDegreeEmbedding for processing concatenated edge features
            	Controls complexity of edge feature integration
        
        edge_degree_projector_size (int, default=64):
            Hidden layer size in edge degree projector MLP.
            	Used by: EdgeDegreeEmbedding for edge feature projection
            	Width of hidden layers in edge processing MLP
        
        # Topology Gradient Parameters
        use_topology_gradients (bool, default=True):
            Whether to use topology gradients for L1 feature enhancement.
            	Used by: _init_topology_mixing(), _compute_topology_gradients(), forward()
            	Incorporates molecular constraint gradients into L=1 spherical harmonics
        
        topology_gradient_clip (float, default=100.0):
            Gradient clipping value for topology gradient computation.
            	Used by: compute_positional_topology_gradients()
            	Prevents exploding gradients from molecular constraint violations
        
        # Legacy Parameters (Maintained for Compatibility)
        regress_forces (bool, default=True):
            IGNORED: Forces regression handled by separate head modules.
            	Legacy parameter from original Equiformer
            	Warning issued if non-default value provided
        
        otf_graph (bool, default=True):
            IGNORED: Uses precomputed graphs from BatchProteinData.
            	Legacy parameter for on-the-fly graph construction
            	This model expects precomputed molecular graphs
        
        use_pbc (bool, default=True):
            IGNORED: Periodic boundary conditions handled in data preprocessing.
            	Legacy parameter for PBC handling
            	PBC should be handled during graph construction
        
        use_pbc_single (bool, default=False):
            IGNORED: PBC single processing handled in data preprocessing.
            	Legacy parameter for single-image PBC
            	Not used in current implementation
        
        max_neighbors (int, default=500):
            IGNORED: Max neighbors constraint applied during graph preprocessing.
            	Legacy parameter for neighbor limiting
            	Should be applied during graph construction phase
        
        max_num_elements (int, default=90):
            IGNORED: Use feature_vocab_sizes["element"] instead.
            	Legacy parameter for element vocabulary size
            	Replaced by more flexible feature_vocab_sizes dictionary
        
        use_atom_edge_embedding (bool, default=True):
            IGNORED: Replaced by enhanced molecular feature embedding.
            	Legacy parameter for atom-edge embedding
            	Superseded by enhanced NodeEmbedder and EdgeDegreeEmbedding
        
        share_atom_edge_embedding (bool, default=False):
            IGNORED: Replaced by enhanced molecular feature embedding.
            	Legacy parameter for embedding sharing
            	Not applicable with new embedding architecture
        
        # Normalization Parameters
        avg_num_nodes (float, default=77.81317):
            Average number of nodes for normalization (from IS2RE dataset statistics).
            	Used by: EdgeDegreeEmbedding for rescaling aggregated edge features
            	Helps stabilize training across different graph sizes
        
        avg_degree (float, default=23.395238876342773):
            Average node degree for normalization (from IS2RE dataset statistics).
            	Used by: EdgeDegreeEmbedding rescale_factor for edge aggregation
            	Prevents features from growing with node degree
    
    Attributes:
        # Core Components
        node_embedder (NodeEmbedder): Embeds atom features to sphere_channels_all
        so3_scalar_embedder (SO3ScalarEmbedder): Converts to SO3 l=0 embedding
        edge_deg_embedding (EdgeDegreeEmbedding): Enhanced edge embedding with molecular features
        blocks (nn.ModuleList): List of TransBlockV2WithEdges transformer layers
        distance_expansion (Union[GaussianSmearing, GaussianRadialBasisLayer]): Distance basis functions
        
        # SO3 Components
        SO3_rotation (nn.ModuleList): SO3_Rotation modules for Wigner-D matrices
        mappingReduced (CoefficientMappingModule): l,m coefficient mapping
        SO3_grid (ModuleListInfo): Grid representations for spherical harmonics
        
        # Normalization
        norm_1 (nn.Module): Input normalization layer
        norm_final (nn.Module): Output normalization layer
        
        # Optional Components
        topology_mixer (SO3_L1_LinearMixing): Mixes topology gradients into L1 features (if use_topology_gradients)
        film (SO3EquivariantFiLM): Time-dependent feature modulation (if use_time)
    
    Input Requirements:
        data (BatchProteinData): Must contain:
            - positions: Atom coordinates [N, 3]
            - element: Atomic numbers [N]
            - edge_index: Graph edges [2, E] 
            - distances: Edge distances [E, 1]
            - distance_vec: Edge vectors [E, 3]
            - Molecular features specified in atom_features and bond_features
            - topology: Dict with bonds, bond_lengths, chirals, planars (if use_topology_gradients)
            - time: Time values [batch_size] (if use_time)
            - batch: Batch assignment [N] (if use_time)
    
    Returns:
        dict: Contains:
            - "node_embedding": SO3_Embedding with final node representations
            - "graph": GraphInfo object with graph metadata for downstream heads
    """
    
    def __init__(
        self,
        # Core architecture parameters
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: Optional[List[int]] = None,
        mmax_list: Optional[List[int]] = None,
        grid_resolution: Optional[int] = None,
        num_sphere_samples: int = 128,
        edge_channels_list: Optional[List[int]] = None,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 128,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        weight_init: str = "uniform",
        max_radius: float = 5.0,  # Used for distance_expansion range
        use_time: bool = False, # whether to expect time tensors and use FiLM to modulate the model
        film_time_embedding_dim: int = 128,  # Dimension for time embedding in FiLM eg output of nonlinear embedding
        film_hidden_dim: int = 256,  # Hidden dimension for FiLM MLP
        film_mlp_layers: int = 2,  # Number of layers in FiLM MLP
        film_num_gaussians: int = 512,  # Number of gaussian basis functions for time smearing
        film_basis_function: str = "gaussian",  # Type of basis function for FiLM time embedding
        # Enhanced molecular feature parameters
        feature_vocab_sizes: Optional[Dict[str, int]] = None,
        atom_features: Optional[List[str]] = None,
        bond_features: Optional[List[str]] = None,
        edge_degree_projector_hidden_layers: int = 2,
        edge_degree_projector_size: int = 64,
        embedding_dim: int = 32, # for tokens to linear embedding
        use_topology_gradients: bool = True,
        topology_gradient_clip: float = 100.0,
        # Legacy parameters (ignored with warnings)
        regress_forces: bool = True,
        otf_graph: bool = True,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        max_neighbors: int = 500,
        max_num_elements: int = 90,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        # previously global params
        avg_num_nodes: float = _AVG_NUM_NODES,
        avg_degree: float = _AVG_DEGREE,

        **kwargs  # Catch any other legacy parameters
    ):
        super().__init__()
        
        # Store core architecture parameters
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        self.lmax_list = lmax_list or [6]
        self.grid_resolution = grid_resolution
        self.num_sphere_samples = num_sphere_samples
        self.num_distance_basis = num_distance_basis
        self.edge_channels_list = edge_channels_list or [self.num_distance_basis, self.sphere_channels, self.sphere_channels]
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.weight_init = weight_init
        self.max_radius = max_radius
        self.use_time = use_time
        self.film_time_embedding_dim = film_time_embedding_dim
        self.film_hidden_dim = film_hidden_dim
        self.film_mlp_layers = film_mlp_layers
        self.film_num_gaussians = film_num_gaussians
        self.film_basis_function = film_basis_function

        
        # Store enhanced feature parameters
        self.feature_vocab_sizes = feature_vocab_sizes or {}
        self.atom_features = atom_features or ['element']
        self.bond_features = bond_features
        self.embedding_dim = embedding_dim
        self.edge_degree_projector_hidden_layers = edge_degree_projector_hidden_layers
        self.edge_degree_projector_size = edge_degree_projector_size
        self.use_topology_gradients = use_topology_gradients
        self.topology_gradient_clip = topology_gradient_clip
        
        # Computed properties (same as original EquiformerV2)
        self.num_resolutions = len(self.lmax_list)
        if self.num_resolutions > 1:
            raise ValueError("EquiformerWEdgesBackbone currently supports only a single resolution (lmax_list must have length 1), really don't know what would happen if you try to use multiple resolutions.")
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.avg_num_nodes = avg_num_nodes
        self.avg_degree = avg_degree
        
        # Warn about ignored parameters
        self._warn_ignored_parameters(
            regress_forces=regress_forces,
            otf_graph=otf_graph,
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            max_neighbors=max_neighbors,
            max_num_elements=max_num_elements,
            use_atom_edge_embedding=use_atom_edge_embedding,
            share_atom_edge_embedding=share_atom_edge_embedding,
            mmax_list=mmax_list,  # Use provided mmax_list or derive from lmax_list
            **kwargs
        )
        
        # Initialize all components
        self._init_so3_components()
        self._init_distance_expansion()
        self._init_enhanced_embeddings()
        self._init_normalization_layers()
        self._init_enhanced_transformer_blocks()
        self._init_film()
        
        if self.use_topology_gradients:
            self._init_topology_mixing()
        
        # Apply weight initialization
        self.apply(partial(eqv2_init_weights, weight_init=self.weight_init))
    
    def _warn_ignored_parameters(self, **kwargs):
        """Warn about parameters that are ignored in EquiformerWEdgesBackbone."""
        ignored_params = {
            'regress_forces': 'Forces regression is handled by separate head modules',
            'otf_graph': 'On-the-fly graph construction not used (precomputed graphs expected)',
            'use_pbc': 'Periodic boundary conditions handled in data preprocessing',
            'use_pbc_single': 'PBC single processing handled in data preprocessing', 
            'max_neighbors': 'Max neighbors constraint applied during graph preprocessing',
            'max_num_elements': 'Element vocabulary determined by feature_vocab_sizes["element"]',
            'use_atom_edge_embedding': 'Atom-edge embedding replaced by enhanced molecular feature embedding',
            'share_atom_edge_embedding': 'Embedding sharing replaced by enhanced molecular feature embedding',
            'mmax_list': 'Mmax list is now derived from lmax list'
        }
        
        for param_name, reason in ignored_params.items():
            if param_name in kwargs:
                param_value = kwargs[param_name]
                # Check if parameter has non-default value that might indicate user expects it to work
                default_values = {
                    'regress_forces': True,
                    'otf_graph': True, 
                    'use_pbc': True,
                    'use_pbc_single': False,
                    'max_neighbors': 500,
                    'max_num_elements': 90,
                    'use_atom_edge_embedding': True,
                    'share_atom_edge_embedding': False,
                    'mmax_list': None
                }
                
                if param_value != default_values.get(param_name):
                    logging.warning(
                        f"Parameter '{param_name}={param_value}' is ignored in EquiformerWEdgesBackbone. "
                        f"Reason: {reason}"
                    )
        
        # Warn about any unknown kwargs
        for param_name in kwargs:
            if param_name not in ignored_params:
                logging.warning(f"Unknown parameter '{param_name}' ignored in EquiformerWEdgesBackbone")

        # set them any way for compatibility
        for param_name in ignored_params:
            setattr(self, param_name, kwargs.get(param_name, None))
    
    def _init_so3_components(self):
        """Initialize SO3 components (same as EquiformerV2Backbone)."""
        # Coefficient mapping
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.lmax_list)
        
        # SO3 rotation modules
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))
        
        # SO3 grid
        self.SO3_grid = ModuleListInfo(f"({max(self.lmax_list)}, {max(self.lmax_list)})")
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)
    
    def _init_distance_expansion(self):
        """Initialize distance expansion module."""
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                start=0.0,
                stop=self.max_radius,
                num_gaussians=self.num_distance_basis,
                basis_width_scalar=2.0
            )
        elif self.distance_function == "gaussian_rbf":
            self.distance_expansion = GaussianRadialBasisLayer(
                num_basis=self.num_distance_basis,
                cutoff=self.max_radius,
            )
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_function}")
    
    def _init_enhanced_embeddings(self):
        """Initialize enhanced embedding modules for molecular features."""
        
        # Enhanced node embedder for atom features
        self.node_embedder = NodeEmbedder(
            feature_vocab_sizes=self.feature_vocab_sizes,
            atom_features=self.atom_features,
            output_dim=self.sphere_channels_all,
            embedding_dim=self.embedding_dim,
        )
        
        # SO3 scalar embedder to convert node embeddings to SO3 format
        self.so3_scalar_embedder = SO3ScalarEmbedder(
            lmax_list=self.lmax_list,
            sphere_channels=self.sphere_channels,
        )
        
        # Enhanced edge degree embedding with molecular features
        self.edge_deg_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax_list=self.lmax_list,
            mmax_list=self.lmax_list,
            SO3_rotation=self.SO3_rotation,
            mappingReduced=self.mappingReduced,
            radial_basis_size=self.num_distance_basis,
            feature_vocab_sizes=self.feature_vocab_sizes,
            use_edge_features=len(self.bond_features) > 0,
            bond_features=self.bond_features,
            use_node_features=len(self.atom_features) > 0,
            node_features=self.atom_features,
            embedding_dim=self.embedding_dim,
            embedding_use_bias=True,
            projector_hidden_layers=self.edge_degree_projector_hidden_layers,
            projector_size=self.edge_degree_projector_size,
            rescale_factor=self.avg_degree,
        )
    
    def _init_normalization_layers(self):
        """Initialize normalization layers."""
        self.norm_1 = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels_all
        )
        
        self.norm_final = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels_all
        )
    
    def _init_enhanced_transformer_blocks(self):
        """Initialize transformer blocks with edge-aware versions."""
        
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2WithEdges(
                sphere_channels=self.sphere_channels,
                attn_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                output_channels=self.sphere_channels,
                lmax_list=self.lmax_list,
                mmax_list=self.lmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=self.edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                # Enhanced edge information parameters
                use_edge_information=len(self.bond_features) > 0 or len(self.atom_features) > 0,
                radial_basis_size=self.num_distance_basis,
                feature_vocab_sizes=self.feature_vocab_sizes,
                use_edge_features=len(self.bond_features) > 0,
                bond_features=self.bond_features,
                use_node_features=len(self.atom_features) > 0,
                node_features=self.atom_features,
                embedding_dim=self.embedding_dim,
                embedding_use_bias=True,
                # Other transformer parameters
                attn_activation=self.attn_activation,
                use_s2_act_attn=self.use_s2_act_attn,
                use_attn_renorm=self.use_attn_renorm,
                ffn_activation=self.ffn_activation,
                use_gate_act=self.use_gate_act,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop,
                use_time=self.use_time,
            )
            self.blocks.append(block)
    
    def _init_topology_mixing(self):
        """Initialize topology gradient mixing for L1 features."""
        self.topology_mixer = SO3_L1_LinearMixing(
            in_channels_list=[self.sphere_channels_all, 3],  # SO3 L1 + topology gradients
            out_channels=self.sphere_channels_all
        )

    def _init_film(self):
        if not self.use_time:
            self.film = None
            return
        else:
            self.film = SO3EquivariantFiLM(
                lmax_list=self.lmax_list,
                mmax_list=self.lmax_list,
                num_channels=self.sphere_channels_all,
                num_layers=self.num_layers,
                time_embedding_dim=self.film_time_embedding_dim,
                hidden_dim=self.film_hidden_dim,
                mlp_layers=self.film_mlp_layers,
                num_gaussians=self.film_num_gaussians,
                basis_function=self.film_basis_function,
                basis_start=0.0,
                basis_end=1.0,
            )
    
    def _extract_feature_dict(self, data: 'BatchProteinData') -> Dict[str, torch.Tensor]:
        """Extract molecular features from BatchProteinData into dictionary format."""
        feature_dict = {}
        
        # Extract atom features
        for feature in self.atom_features:
            if hasattr(data, feature):
                feature_dict[feature] = getattr(data, feature)
            else:
                raise ValueError(f"Atom feature {feature} not found in batch data")
        
        # Extract bond features  
        for feature in self.bond_features:
            if hasattr(data, feature):
                feature_dict[feature] = getattr(data, feature)
            else:
                raise ValueError(f"Bond feature {feature} not found in batch data")
        
        return feature_dict
    
    def _compute_topology_gradients(
        self, 
        data: 'BatchProteinData'
    ) -> Optional[torch.Tensor]:
        """Compute topology gradients for L1 features."""
        if not self.use_topology_gradients:
            return None
            
        # Create atom mask for any masked atoms
        masked_elements = None
        if data.atom_masked_mask is not None:
            masked_elements = data.atom_masked_mask
        
        atom_mask = None if masked_elements is None else ~masked_elements
        
        # Extract topology information
        topology = data.topology if hasattr(data, 'topology') else {}
        bonds = topology.get('bonds', torch.empty(0, 2, dtype=torch.long, device=data.positions.device))
        bond_lengths = topology.get('bond_lengths', torch.empty(0, 1, device=data.positions.device))
        chirals = topology.get('chirals', torch.empty(0, 4, dtype=torch.long, device=data.positions.device))
        planars = topology.get('planars', torch.empty(0, 4, dtype=torch.long, device=data.positions.device))
        
        # Compute gradients
        gradients = compute_positional_topology_gradients(
            r=data.positions,
            bond_indexes=bonds,
            bond_lengths=bond_lengths,
            chirals=chirals,
            planars=planars,
            gclip=self.topology_gradient_clip,
            atom_mask=atom_mask
        )
        
        return gradients
    
    def forward(self, data: 'BatchProteinData'):
        """
        Forward pass with enhanced molecular feature processing.
        
        Args:
            data: BatchProteinData containing precomputed graphs and molecular features
            
        Returns:
            dict: Dictionary containing node embeddings and graph information
        """

        # check feature sizes against vocab
        for atom_feature_name in self.atom_features:
            if atom_feature_name not in self.feature_vocab_sizes:
                raise ValueError(f"Atom feature '{atom_feature_name}' not found in feature_vocab_sizes")
            if getattr(data, atom_feature_name) is None or getattr(data, atom_feature_name).max() >= self.feature_vocab_sizes[atom_feature_name]:
                raise ValueError(f"Atom feature '{atom_feature_name}' exceeds vocabulary size {self.feature_vocab_sizes[atom_feature_name]}")
            
        for bond_feature_name in self.bond_features:
            if bond_feature_name not in self.feature_vocab_sizes:
                raise ValueError(f"Bond feature '{bond_feature_name}' not found in feature_vocab_sizes")
            if getattr(data, bond_feature_name) is None or getattr(data, bond_feature_name).max() >= self.feature_vocab_sizes[bond_feature_name]:
                raise ValueError(f"Bond feature '{bond_feature_name}' exceeds vocabulary size {self.feature_vocab_sizes[bond_feature_name]}")

        atomic_numbers = data.element
        num_atoms = len(atomic_numbers)
        
        # Extract molecular features
        feature_dict = self._extract_feature_dict(data)
        
        # Compute edge rotation matrices for SO3 operations
        edge_index = data.edge_index
        edge_distance = data.distances
        edge_distance_vec = data.distance_vec
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec).to(data.positions.dtype)
        
        
        # Set up SO3 rotation matrices
        for i, SO3_rot in enumerate(self.SO3_rotation):
            SO3_rot.set_wigner(edge_rot_mat) # NOTE: This forces the SO3_rotation to be in float32 regardless of dtype of data, fix that now
            SO3_rot.wigner = SO3_rot.wigner.to(data.positions.dtype).detach()  # Ensure wigner is in correct dtype
            SO3_rot.wigner_inv = SO3_rot.wigner_inv.to(data.positions.dtype).detach()  # Ensure inverse wigner is in correct dtype
        
        # Distance embedding using enhanced radial basis
        if self.distance_function == "gaussian":
            edge_distance_rbf = self.distance_expansion(edge_distance)
        elif self.distance_function == "gaussian_rbf":
            # Gaussian Radial Basis Layer returns a tensor directly
            edge_distance_rbf = self.distance_expansion(edge_distance.squeeze(-1))  # Ensure correct shape for RBF layer
        
        ###############################################################
        # Initialize node embeddings
        ###############################################################
        
        # Enhanced node embedding with molecular features
        node_attributes = self.node_embedder(feature_dict)
        
        # Convert to SO3 embedding format (L=0 coefficients)
        node_embedding = self.so3_scalar_embedder(node_attributes)
        
        ###############################################################
        # Add edge information to node embeddings
        ###############################################################
        
        # Enhanced edge degree embedding with molecular features
        edge_embedding = self.edge_deg_embedding(
            edge_distance_rbf=edge_distance_rbf,
            edge_index=edge_index,
            num_nodes=num_atoms,
            feature_dict=feature_dict,
            node_offset=0
        )
        
        # Add edge information to node embeddings
        node_embedding.embedding = node_embedding.embedding + edge_embedding.embedding
        
        # Apply input normalization
        node_embedding.embedding = self.norm_1(node_embedding.embedding)
        
        ###############################################################
        # Mix topology gradients into L1 features (if enabled)
        ###############################################################
        
        if self.use_topology_gradients:
            topology_gradients = self._compute_topology_gradients(data)
            if topology_gradients is not None:
                # Extract current L1 features
                current_l1_features = node_embedding.embedding[:, 1:4, :]  # L=1, m=-1,0,1
                
                # Mix with topology gradients
                mixed_l1_features = self.topology_mixer([current_l1_features, topology_gradients])
                
                # Update L1 features in embedding
                node_embedding.embedding[:, 1:4, :] = mixed_l1_features

        ###############################################################
        # Apply FiLM modulation if time is provided
        ###############################################################
        if self.use_time:
            assert data.time is not None, "Time tensor must be provided for FiLM modulation"
            assert data.batch is not None, "Batch tensor must be provided for FiLM modulation"
            # Get FiLM modulation coefficients per spherical harmonic per layer
            time_coefficient_weights = self.film(data.time) # [N layers, batch_size, (lmax + 1)^2, d]
            
            # compute the norms for FiLM coefficients and expose them in case we want to apply l2 norm to the loss
            film_norm = torch.mean(
                time_coefficient_weights ** 2
            )
        else:
            film_norm = None

        ###############################################################
        # Transformer blocks with enhanced edge information
        ###############################################################
        
        for i in range(self.num_layers):
            film_coefs = None
            if self.use_time:
                # Get FiLM coefficients for this layer
                film_coefs = time_coefficient_weights[i]
            node_embedding = self.blocks[i](
                x=node_embedding,
                edge_distance=edge_distance_rbf,
                edge_index=edge_index,
                feature_dict=feature_dict,
                film_coefs=film_coefs,
                batch=data.batch if hasattr(data, 'batch') else None,
                node_offset=0,
            )
        
        # Final output normalization
        node_embedding.embedding = self.norm_final(node_embedding.embedding)
        
        ###############################################################
        # Return embeddings in format compatible with heads
        ###############################################################
        
        # Create graph information object (similar to original)
        graph_info = type('GraphInfo', (), {
            'atomic_numbers': atomic_numbers,
            'atomic_numbers_full': atomic_numbers,  # For compatibility
            'edge_distance': edge_distance_rbf,
            'edge_index': edge_index,
            'node_offset': 0,
        })()
        
        return {
            "node_embedding": node_embedding,
            "film_norm": film_norm,
            "graph": graph_info,
        }