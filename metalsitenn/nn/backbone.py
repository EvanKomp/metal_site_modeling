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

from .embeddings import NodeEmbedder, SO3ScalarEmbedder, EdgeDegreeEmbedding
from .topology import compute_positional_topology_gradients, SO3_L1_LinearMixing
from .attention import TransBlockV2WithEdges

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
    
    Designed to accept BatchProteinData with:
    - Precomputed molecular graphs with rich atom/bond features
    - Topology information for L1 gradient computation
    - Enhanced edge information processing in attention layers
    
    Args:
        num_layers: Number of transformer layers
        sphere_channels: Number of spherical channels (one set per resolution)
        attn_hidden_channels: Number of hidden channels used during SO(2) graph attention
        num_heads: Number of attention heads
        attn_alpha_channels: Number of channels for alpha vector in each attention head
        attn_value_channels: Number of channels for value vector in each attention head
        ffn_hidden_channels: Number of hidden channels used during feedforward network
        norm_type: Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])
        lmax_list: List of degrees (l) for each resolution
        mmax_list: List of orders (m) for each resolution
        grid_resolution: Resolution of SO3_grid
        num_sphere_samples: Number of samples used to approximate the integration of the sphere in the output blocks
        edge_channels_list: List of sizes of invariant edge embedding
        use_m_share_rad: Whether all m components within a type-L vector share radial function weights
        distance_function: Type of distance function (['gaussian', 'gaussian_rbf'])
        num_distance_basis: Number of distance basis functions
        attn_activation: Type of activation function for SO(2) graph attention
        use_s2_act_attn: Whether to use attention after S2 activation
        use_attn_renorm: Whether to re-normalize attention weights
        ffn_activation: Type of activation function for feedforward network
        use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp: If `True`, use projecting to grids and performing MLPs for FFN
        use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False
        alpha_drop: Dropout rate for attention weights
        drop_path_rate: Drop path rate
        proj_drop: Dropout rate for outputs of attention and FFN
        weight_init: Weight initialization method
        max_radius: Used for distance_expansion range
        
        # Enhanced molecular feature parameters
        feature_vocab_sizes: Vocabulary sizes for molecular features
        atom_features: List of atom feature names to use
        bond_features: List of bond feature names to use
        embedding_dim: Embedding dimension for molecular features
        use_topology_gradients: Whether to use topology gradients for L1 features
        topology_gradient_clip: Gradient clipping value for topology gradients
        
        # Legacy parameters (ignored with warnings)
        regress_forces: IGNORED - Forces regression handled by separate head modules
        otf_graph: IGNORED - Uses precomputed graphs from BatchProteinData
        use_pbc: IGNORED - PBC handled in data preprocessing
        use_pbc_single: IGNORED - PBC handled in data preprocessing
        max_neighbors: IGNORED - Applied during graph preprocessing
        max_num_elements: IGNORED - Use feature_vocab_sizes["element"] instead
        use_atom_edge_embedding: IGNORED - Replaced by enhanced molecular features
        share_atom_edge_embedding: IGNORED - Replaced by enhanced molecular features
        avg_num_nodes: Average number of nodes for normalization
        avg_degree: Average degree for normalization
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
        num_distance_basis: int = 512,
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
        # Enhanced molecular feature parameters
        feature_vocab_sizes: Optional[Dict[str, int]] = None,
        atom_features: Optional[List[str]] = None,
        bond_features: Optional[List[str]] = None,
        embedding_dim: int = 32,
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
        self.mmax_list = mmax_list or [2]
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
        
        # Store enhanced feature parameters
        self.feature_vocab_sizes = feature_vocab_sizes or {}
        self.atom_features = atom_features or ['element', 'charge', 'nhyd', 'hyb']
        self.bond_features = bond_features or ['bond_order', 'is_in_ring', 'is_aromatic']
        self.embedding_dim = embedding_dim
        self.use_topology_gradients = use_topology_gradients
        self.topology_gradient_clip = topology_gradient_clip
        
        # Computed properties (same as original EquiformerV2)
        self.num_resolutions = len(self.lmax_list)
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
            **kwargs
        )
        
        # Initialize all components
        self._init_so3_components()
        self._init_distance_expansion()
        self._init_enhanced_embeddings()
        self._init_normalization_layers()
        self._init_enhanced_transformer_blocks()
        
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
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)
        
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
                basis_width_scalar=(self.max_radius - 0.0) / self.num_distance_basis
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
            mmax_list=self.mmax_list,
            SO3_rotation=self.SO3_rotation,
            mappingReduced=self.mappingReduced,
            radial_basis_size=self.num_distance_basis,
            feature_vocab_sizes=self.feature_vocab_sizes,
            use_edge_features=True,
            bond_features=self.bond_features,
            use_node_features=True,
            node_features=self.atom_features,
            embedding_dim=self.embedding_dim,
            embedding_use_bias=True,
            projector_hidden_layers=2,
            projector_size=64,
            rescale_factor=1.0,
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
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=self.edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                # Enhanced edge information parameters
                use_edge_information=True,
                radial_basis_size=self.num_distance_basis,
                feature_vocab_sizes=self.feature_vocab_sizes,
                use_edge_features=True,
                bond_features=self.bond_features,
                use_node_features=True,
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
            )
            self.blocks.append(block)
    
    def _init_topology_mixing(self):
        """Initialize topology gradient mixing for L1 features."""
        self.topology_mixer = SO3_L1_LinearMixing(
            in_channels_list=[self.sphere_channels_all, 3],  # SO3 L1 + topology gradients
            out_channels=self.sphere_channels_all
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
        # TODO: check for masked token
        
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
        atomic_numbers = data.element
        num_atoms = len(atomic_numbers)
        
        # Extract molecular features
        feature_dict = self._extract_feature_dict(data)
        
        # Compute edge rotation matrices for SO3 operations
        edge_index = data.edge_index
        edge_distance = data.distances
        edge_distance_vec = data.distance_vec
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        
        # Set up SO3 rotation matrices
        for i, SO3_rot in enumerate(self.SO3_rotation):
            SO3_rot.set_wigner(edge_rot_mat)
        
        # Distance embedding using enhanced radial basis
        edge_distance_rbf = self.distance_expansion(edge_distance)
        
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
        # Transformer blocks with enhanced edge information
        ###############################################################
        
        for i in range(self.num_layers):
            node_embedding = self.blocks[i](
                x=node_embedding,
                edge_distance=edge_distance_rbf,
                edge_index=edge_index,
                feature_dict=feature_dict,
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
            "graph": graph_info,
        }