# metalsitenn/nn/attention.py
'''
* Author: Evan Komp
* Created: 8/13/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT


This module provides SO2 equivariant graph attention layers that incorporate
rich molecular features (distance, node attributes, edge attributes) and
complete transformer blocks for enhanced molecular representation learning.
'''

import copy
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch_geometric

from fairchem.core.common import gp_utils
from fairchem.core.models.equiformer_v2.activation import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from fairchem.core.models.equiformer_v2.drop import (
    EquivariantDropoutArraySphericalHarmonics, 
    GraphDropPath
)
from fairchem.core.models.equiformer_v2.layer_norm import get_normalization_layer
from fairchem.core.models.equiformer_v2.radial_function import RadialFunction
from fairchem.core.models.equiformer_v2.so2_ops import SO2_Convolution
from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from fairchem.core.models.equiformer_v2.transformer_block import FeedForwardNetwork

from .embeddings import EdgeProjector


class SO2EquivariantGraphAttentionWEdgesV2(nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels: Number of spherical channels
        hidden_channels: Number of hidden channels used during the SO(2) conv
        num_heads: Number of attention heads
        attn_alpha_channels: Number of channels for alpha vector in each attention head
        attn_value_channels: Number of channels for value vector in each attention head
        output_channels: Number of output channels
        lmax_list: List of degrees (l) for each resolution
        mmax_list: List of orders (m) for each resolution
        SO3_rotation: Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced: Class to convert l and m indices once node embedding is rotated
        SO3_grid: Class used to convert from grid the spherical harmonic representations
        edge_channels_list: List of sizes of invariant edge embedding
        use_m_share_rad: Whether all m components within a type-L vector of one channel share radial function weights
        
        # EdgeProjector parameters
        use_edge_information: Whether to use edge information in the attention mechanism
        radial_basis_size: Number of radial basis functions expected
        feature_vocab_sizes: Dictionary mapping feature names to vocab sizes
        use_edge_features: Whether to use edge features
        bond_features: List of bond feature names to use if using any
        use_node_features: Whether to use node features
        node_features: List of node feature names to use if using any
        embedding_dim: Embedding dimension for node and edge features
        embedding_use_bias: Whether to use bias in the embedding layers
        
        activation: Type of activation function
        use_s2_act_attn: Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm: Whether to re-normalize attention weights
        use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False.
        alpha_drop: Dropout rate for attention weights
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        output_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        edge_channels_list: List[int],
        use_m_share_rad: bool = False,
        # EdgeProjector parameters
        use_edge_information: bool = True,
        radial_basis_size: int = 50,
        feature_vocab_sizes: Optional[Dict[str, int]] = None,
        use_edge_features: bool = True,
        bond_features: Optional[List[str]] = None,
        use_node_features: bool = True,
        node_features: Optional[List[str]] = None,
        embedding_dim: int = 32,
        embedding_use_bias: bool = True,
        activation: str = "scaled_silu", # does nothing, maybe FFNN was attached to this at one point
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        # Edge feature processing
        self.use_edge_information = use_edge_information
        self.use_m_share_rad = use_m_share_rad
        
        if feature_vocab_sizes is None:
            feature_vocab_sizes = {}
        if bond_features is None:
            bond_features = ['bond_order', 'is_in_ring', 'is_aromatic']
        if node_features is None:
            node_features = ['element', 'charge', 'nhyd', 'hyb']

        # Initialize edge projector
        if self.use_edge_information:
            self.edge_projector = EdgeProjector(
                radial_basis_size=radial_basis_size,
                feature_vocab_sizes=feature_vocab_sizes,
                use_edge_features=use_edge_features,
                bond_features=bond_features,
                use_node_features=use_node_features,
                node_features=node_features,
                output_dim=edge_channels_list[-1],  # Match expected output dimension
                embedding_dim=embedding_dim,
                embedding_use_bias=embedding_use_bias,
                use_projector=False,  # Just concatenation, no radial function
                projector_hidden_layers=1,
                projector_size=64
            )
            # Update edge channels list input size based on projector concatenated size
            self.edge_channels_list = copy.deepcopy(edge_channels_list)
            self.edge_channels_list[0] = self.edge_projector.input_size
        else:
            self.edge_projector = None
            self.edge_channels_list = copy.deepcopy(edge_channels_list)

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = [
                *self.edge_channels_list,
                2 * self.sphere_channels * (max(self.lmax_list) + 1),
            ]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_LinearV2(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
            lmax=self.lmax_list[0],
        )

    def forward(
        self,
        x: SO3_Embedding,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        feature_dict: Optional[Dict[str, torch.Tensor]] = None,
        node_offset: int = 0,
    ):
        """
        Forward pass through SO2EquivariantGraphAttention.
        
        Args:
            x: SO3_Embedding node features
            edge_distance: [E, radial_basis_size] radial basis encoded distances
            edge_index: [2, E] edge connectivity
            feature_dict: Dictionary of additional node/edge features for EdgeProjector
            node_offset: Node offset for distributed computing
            
        Returns:
            SO3_Embedding: Updated node embeddings
        """
        # Compute edge scalar features (invariant to rotations)
        if self.use_edge_information:
            if feature_dict is None:
                feature_dict = {}
            x_edge = self.edge_projector(edge_distance, edge_index, feature_dict)
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region(x.embedding, dim=0)
            x_source.set_embedding(x_full)
            x_target.set_embedding(x_full)
        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1,
                x_alpha_num_channels,
                x_0_extra.shape[1] - x_alpha_num_channels,
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )  # for activation
                x_0_alpha = x_0_extra.narrow(
                    1, 0, x_alpha_num_channels
                )  # for attention weights
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[1] - node_offset, len(x.embedding))

        # Project
        return self.proj(x_message)


class TransBlockV2WithEdges(nn.Module):
    """
    Updated TransBlockV2 that leverages SO2EquivariantGraphAttentionWEdgesV2 
    for enhanced edge information processing.

    Args:
        sphere_channels: Number of spherical channels
        attn_hidden_channels: Number of hidden channels used during SO(2) graph attention
        num_heads: Number of attention heads
        attn_alpha_channels: Number of channels for alpha vector in each attention head
        attn_value_channels: Number of channels for value vector in each attention head
        ffn_hidden_channels: Number of hidden channels used during feedforward network
        output_channels: Number of output channels
        lmax_list: List of degrees (l) for each resolution
        mmax_list: List of orders (m) for each resolution
        SO3_rotation: Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced: Class to convert l and m indices once node embedding is rotated
        SO3_grid: Class used to convert from grid the spherical harmonic representations
        edge_channels_list: List of sizes of invariant edge embedding
        use_m_share_rad: Whether all m components within a type-L vector share radial weights
        
        # EdgeProjector parameters for enhanced edge information
        use_edge_information: Whether to use edge information in the attention mechanism
        radial_basis_size: Number of radial basis functions expected
        feature_vocab_sizes: Dictionary mapping feature names to vocab sizes
        use_edge_features: Whether to use edge features
        bond_features: List of bond feature names to use if using any
        use_node_features: Whether to use node features in edge projector
        node_features: List of node feature names to use if using any
        embedding_dim: Embedding dimension for node and edge features
        embedding_use_bias: Whether to use bias in the embedding layers
        
        attn_activation: Type of activation function for SO(2) graph attention
        use_s2_act_attn: Whether to use attention after S2 activation
        use_attn_renorm: Whether to re-normalize attention weights
        ffn_activation: Type of activation function for feedforward network
        use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp: If `True`, use projecting to grids and performing MLPs for FFN
        use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False
        norm_type: Type of normalization layer (['layer_norm', 'layer_norm_sh'])
        alpha_drop: Dropout rate for attention weights
        drop_path_rate: Drop path rate
        proj_drop: Dropout rate for outputs of attention and FFN
    """

    def __init__(
        self,
        sphere_channels: int,
        attn_hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        output_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        edge_channels_list: List[int],
        use_m_share_rad: bool = False,
        # EdgeProjector parameters
        use_edge_information: bool = True,
        radial_basis_size: int = 50,
        feature_vocab_sizes: Optional[Dict[str, int]] = None,
        use_edge_features: bool = True,
        bond_features: Optional[List[str]] = None,
        use_node_features: bool = True,
        node_features: Optional[List[str]] = None,
        embedding_dim: int = 32,
        embedding_use_bias: bool = True,
        # Other parameters matching original TransBlockV2
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        norm_type: str = "rms_norm_sh",
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        # Set defaults for edge projector parameters
        if feature_vocab_sizes is None:
            feature_vocab_sizes = {}
        if bond_features is None:
            bond_features = ['bond_order', 'is_in_ring', 'is_aromatic']
        if node_features is None:
            node_features = ['element', 'charge', 'nhyd', 'hyb']

        max_lmax = max(lmax_list)
        
        # Pre-attention normalization
        self.norm_1 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

        # Enhanced SO2 Graph Attention with edge information
        self.ga = SO2EquivariantGraphAttentionWEdgesV2(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            edge_channels_list=edge_channels_list,
            use_m_share_rad=use_m_share_rad,
            # EdgeProjector parameters
            use_edge_information=use_edge_information,
            radial_basis_size=radial_basis_size,
            feature_vocab_sizes=feature_vocab_sizes,
            use_edge_features=use_edge_features,
            bond_features=bond_features,
            use_node_features=use_node_features,
            node_features=node_features,
            embedding_dim=embedding_dim,
            embedding_use_bias=embedding_use_bias,
            # Attention parameters
            activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        # Dropout and drop path
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = (
            EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False)
            if proj_drop > 0.0
            else None
        )

        # Pre-FFN normalization
        self.norm_2 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels,
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
        )

        # Optional shortcut connection for dimension mismatch
        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(
                sphere_channels, output_channels, lmax=max_lmax
            )
        else:
            self.ffn_shortcut = None

    def forward(
        self,
        x: SO3_Embedding,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        feature_dict: Dict[str, torch.Tensor],
        batch: Optional[torch.Tensor] = None,  # for GraphDropPath
        node_offset: int = 0,
    ):
        """
        Forward pass through the transformer block.
        
        Args:
            x: SO3_Embedding node features
            edge_distance: [E, radial_basis_size] radial basis encoded distances
            edge_index: [2, E] edge connectivity
            feature_dict: Dictionary of additional node/edge features for EdgeProjector
            batch: Batch information for GraphDropPath
            node_offset: Node offset for distributed computing
            
        Returns:
            SO3_Embedding: Updated node embeddings
        """
        output_embedding = x.clone()

        # Store residual connection for attention
        x_res = output_embedding.clone()

        # Pre-attention normalization
        output_embedding.embedding = self.norm_1(output_embedding.embedding)

        # Enhanced attention with edge information
        output_embedding = self.ga(
            x=output_embedding,
            edge_distance=edge_distance,
            edge_index=edge_index,
            feature_dict=feature_dict,
            node_offset=node_offset,
        )

        # Apply projection dropout if specified
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding)

        # Apply drop path if specified
        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )

        # Residual connection for attention
        output_embedding.embedding = output_embedding.embedding + x_res.embedding

        # Store residual connection for FFN
        x_res = output_embedding.clone()

        # Pre-FFN normalization
        output_embedding.embedding = self.norm_2(output_embedding.embedding)

        # Feed-forward network
        output_embedding = self.ffn(output_embedding)

        # Apply projection dropout if specified
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding)

        # Apply drop path if specified
        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )

        # Residual connection for FFN (with optional shortcut for dimension mismatch)
        if self.ffn_shortcut is not None:
            x_res = self.ffn_shortcut(x_res)
        output_embedding.embedding = output_embedding.embedding + x_res.embedding

        return output_embedding