# metalsitenn/nn/pretrained_config.py
'''
* Author: Evan Komp
* Created: 8/20/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import Dict, List, Optional
from transformers.configuration_utils import PretrainedConfig

class EquiformerWEdgesConfig(PretrainedConfig):
    """
    Configuration class for EquiformerWEdgesBackbone model.
    
    This configuration class organizes the large number of parameters into logical groups
    for better maintainability and understanding. All parameters that are tunable in the
    backbone are included, along with additional parameters useful for loss computation
    and training.
    """
    
    model_type = "equiformer_w_edges"
    
    def __init__(
        self,
        # === CORE ARCHITECTURE ===
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 64,
        num_heads: int = 8,
        attn_alpha_channels: int = 64,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 128,
        norm_type: str = "layer_norm_sh",
        lmax_list: List[int] = [3],
        mmax_list: List[int] = None,
        grid_resolution: Optional[int] = 18,
        
        # === DISTANCE & SPATIAL ===
        num_distance_basis: int = 512,
        distance_function: str = "gaussian",
        max_radius: float = 12.0,
        
        # === MOLECULAR FEATURES ===
        feature_vocab_sizes: Dict[str, int] = None,
        atom_features: List[str] = None,
        bond_features: List[str] = None,
        embedding_dim: int = 32,
        edge_degree_projector_hidden_layers: int = 2,
        edge_degree_projector_size: int = 128,
        edge_channels_list: List[int] = None,
        
        # === ATTENTION MECHANISMS ===
        use_m_share_rad: bool = False,
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        
        # === FEED FORWARD NETWORKS ===
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        
        # === REGULARIZATION ===
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        proj_drop: float = 0.0,
        weight_init: str = "uniform",
        
        # === TOPOLOGY GRADIENTS ===
        use_topology_gradients: bool = True,
        topology_gradient_clip: float = 100.0,
        
        # === TIME/FILM MODULATION ===
        use_time: bool = False,
        film_time_embedding_dim: int = 128,
        film_hidden_dim: int = 128,
        film_mlp_layers: int = 2,
        film_num_gaussians: int = 512,
        film_basis_function: str = "gaussian",
        
        # === NORMALIZATION STATISTICS ===
        avg_num_nodes: float = 100, # num atoms in the system
        avg_degree: float = 20, # eg. the graph size
        
        # === TASK-SPECIFIC PARAMETERS ===

        # NODE CLASSIFICATION
        node_class_weights: Optional[List[float]] = None,
        node_class_label_smoothing: float = 0.0,

        # FILM LOSS
        film_l2_loss_weight: float = 0.0,
    ):
        """
        Initialize EquiformerWEdgesConfig.
        
        Args:
            # === CORE ARCHITECTURE ===
            num_layers: Number of transformer layers in the backbone.
            sphere_channels: Number of channels for spherical harmonics representations.
            attn_hidden_channels: Hidden channels in attention mechanism.
            num_heads: Number of attention heads.
            attn_alpha_channels: Alpha channels in attention computation.
            attn_value_channels: Value channels in attention computation.
            ffn_hidden_channels: Hidden channels in feed-forward networks.
            norm_type: Type of normalization layer to use.
            lmax_list: List of maximum spherical harmonic degrees per resolution.
            mmax_list: List of maximum azimuthal quantum numbers (derived from lmax if None).
            grid_resolution: Resolution for SO(3) grid representation.
            
            # === DISTANCE & SPATIAL ===
            num_distance_basis: Number of basis functions for distance expansion.
            distance_function: Type of distance expansion function.
            max_radius: Maximum interaction radius for molecular interactions.
            
            # === MOLECULAR FEATURES ===
            feature_vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
            atom_features: List of atom feature names to use.
            bond_features: List of bond feature names to use.
            embedding_dim: Embedding dimension for categorical features.
            edge_degree_projector_hidden_layers: Hidden layers in edge degree projector.
            edge_degree_projector_size: Hidden size in edge degree projector.
            edge_channels_list: List of edge channels for different resolutions.
            
            # === ATTENTION MECHANISMS ===
            use_m_share_rad: Whether to share radial functions across azimuthal numbers.
            attn_activation: Activation function for attention layers.
            use_s2_act_attn: Whether to use S2 activation in attention.
            use_attn_renorm: Whether to use attention renormalization.
            
            # === FEED FORWARD NETWORKS ===
            ffn_activation: Activation function for feed-forward networks.
            use_gate_act: Whether to use gated activation functions.
            use_grid_mlp: Whether to use grid-based MLPs.
            use_sep_s2_act: Whether to use separate S2 activations.
            
            # === REGULARIZATION ===
            alpha_drop: Dropout rate for alpha channels.
            drop_path_rate: DropPath rate for stochastic depth.
            proj_drop: Dropout rate for projections.
            weight_init: Weight initialization strategy.
            
            # === TOPOLOGY GRADIENTS ===
            use_topology_gradients: Whether to use topology gradient features.
            topology_gradient_clip: Gradient clipping value for topology gradients.
            
            # === TIME/FILM MODULATION ===
            use_time: Whether to use time-dependent features.
            film_time_embedding_dim: Dimension for time embeddings in FiLM.
            film_hidden_dim: Hidden dimension for FiLM layers.
            film_mlp_layers: Number of MLP layers in FiLM.
            film_num_gaussians: Number of Gaussian basis functions for FiLM.
            film_basis_function: Basis function type for FiLM.
            
            # === NORMALIZATION STATISTICS ===
            avg_num_nodes: Average number of nodes for normalization.
            avg_degree: Average node degree for normalization.
            
            # === TASK-SPECIFIC PARAMETERS ===
            node_class_weights: Weights for class balancing in node classification.
            node_class_label_smoothing: Label smoothing factor for node classification.
            film_l2_loss_weight: L2 loss weight for FiLM regularization.
            
        """
        super().__init__()
        # Set default values for lists and dicts
        if lmax_list is None:
            lmax_list = [3]
        if feature_vocab_sizes is None:
            raise ValueError("feature_vocab_sizes must be provided")
        if atom_features is None:
            atom_features = ['element', 'charge', 'nhyd', 'hyb']
        if bond_features is None:
            bond_features = ['bond_order', 'is_in_ring', 'is_aromatic']

        
        # === CORE ARCHITECTURE ===
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution
        
        # === DISTANCE & SPATIAL ===
        self.num_distance_basis = num_distance_basis
        self.distance_function = distance_function
        self.max_radius = max_radius
        
        # === MOLECULAR FEATURES ===
        self.feature_vocab_sizes = feature_vocab_sizes
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.embedding_dim = embedding_dim
        self.edge_degree_projector_hidden_layers = edge_degree_projector_hidden_layers
        self.edge_degree_projector_size = edge_degree_projector_size
        self.edge_channels_list = edge_channels_list
        
        # === ATTENTION MECHANISMS ===
        self.use_m_share_rad = use_m_share_rad
        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        
        # === FEED FORWARD NETWORKS ===
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        # === REGULARIZATION ===
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.weight_init = weight_init
        
        # === TOPOLOGY GRADIENTS ===
        self.use_topology_gradients = use_topology_gradients
        self.topology_gradient_clip = topology_gradient_clip
        
        # === TIME/FILM MODULATION ===
        self.use_time = use_time
        self.film_time_embedding_dim = film_time_embedding_dim
        self.film_hidden_dim = film_hidden_dim
        self.film_mlp_layers = film_mlp_layers
        self.film_num_gaussians = film_num_gaussians
        self.film_basis_function = film_basis_function
        
        # === NORMALIZATION STATISTICS ===
        self.avg_num_nodes = avg_num_nodes
        self.avg_degree = avg_degree
        
        # === TASK-SPECIFIC PARAMETERS ===
        self.node_class_weights = node_class_weights
        self.node_class_label_smoothing = node_class_label_smoothing
        self.film_l2_loss_weight = film_l2_loss_weight
        
    
    @property
    def backbone_kwargs(self) -> Dict:
        """
        Get kwargs dictionary for initializing EquiformerWEdgesBackbone.
        
        Returns:
            Dictionary containing all parameters needed by the backbone.
        """
        return {
            # Core architecture
            'num_layers': self.num_layers,
            'sphere_channels': self.sphere_channels,
            'attn_hidden_channels': self.attn_hidden_channels,
            'num_heads': self.num_heads,
            'attn_alpha_channels': self.attn_alpha_channels,
            'attn_value_channels': self.attn_value_channels,
            'ffn_hidden_channels': self.ffn_hidden_channels,
            'norm_type': self.norm_type,
            'lmax_list': self.lmax_list,
            'mmax_list': self.mmax_list,
            'grid_resolution': self.grid_resolution,
            
            # Distance & spatial
            'num_distance_basis': self.num_distance_basis,
            'distance_function': self.distance_function,
            'max_radius': self.max_radius,
            
            # Molecular features
            'feature_vocab_sizes': self.feature_vocab_sizes,
            'atom_features': self.atom_features,
            'bond_features': self.bond_features,
            'embedding_dim': self.embedding_dim,
            'edge_degree_projector_hidden_layers': self.edge_degree_projector_hidden_layers,
            'edge_degree_projector_size': self.edge_degree_projector_size,
            'edge_channels_list': self.edge_channels_list,
            
            # Attention mechanisms
            'use_m_share_rad': self.use_m_share_rad,
            'attn_activation': self.attn_activation,
            'use_s2_act_attn': self.use_s2_act_attn,
            'use_attn_renorm': self.use_attn_renorm,
            
            # Feed forward networks
            'ffn_activation': self.ffn_activation,
            'use_gate_act': self.use_gate_act,
            'use_grid_mlp': self.use_grid_mlp,
            'use_sep_s2_act': self.use_sep_s2_act,
            
            # Regularization
            'alpha_drop': self.alpha_drop,
            'drop_path_rate': self.drop_path_rate,
            'proj_drop': self.proj_drop,
            'weight_init': self.weight_init,
            
            # Topology gradients
            'use_topology_gradients': self.use_topology_gradients,
            'topology_gradient_clip': self.topology_gradient_clip,
            
            # Time/FiLM modulation
            'use_time': self.use_time,
            'film_time_embedding_dim': self.film_time_embedding_dim,
            'film_hidden_dim': self.film_hidden_dim,
            'film_mlp_layers': self.film_mlp_layers,
            'film_num_gaussians': self.film_num_gaussians,
            'film_basis_function': self.film_basis_function,
            
            # Normalization statistics
            'avg_num_nodes': self.avg_num_nodes,
            'avg_degree': self.avg_degree,
        }
    
    def validate_config(self) -> None:
        """
        Validate configuration parameters for consistency.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate core architecture
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.sphere_channels <= 0:
            raise ValueError("sphere_channels must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
            
        # Validate lmax_list
        if not self.lmax_list or any(l < 0 for l in self.lmax_list):
            raise ValueError("lmax_list must contain non-negative integers")
        if len(self.lmax_list) > 1:
            raise ValueError("Currently only single resolution supported (lmax_list length = 1)")
            
        # Validate molecular features
        if not self.atom_features:
            raise ValueError("atom_features cannot be empty")
        if not self.bond_features:
            raise ValueError("bond_features cannot be empty")
            
        # Check feature vocabulary coverage
        missing_features = []
        for feature in self.atom_features + self.bond_features:
            if feature not in self.feature_vocab_sizes:
                missing_features.append(feature)
        if missing_features:
            raise ValueError(f"Missing vocabulary sizes for features: {missing_features}")
                
        # Validate FiLM parameters
        if self.use_time and self.film_l2_loss_weight < 0:
            raise ValueError("film_l2_loss_weight must be non-negative")
                
        # Validate training parameters
        if self.gradient_clipping <= 0:
            raise ValueError("gradient_clipping must be positive")
        