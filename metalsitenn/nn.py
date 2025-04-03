# metalsitenn/nn.py
'''
* Author: Evan Komp
* Created: 11/4/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Network adapted fron e3nn and equiformer.
https://github.com/e3nn/e3nn
https://github.com/atomicarchitects/equiformer/tree/master
'''
from typing import List, Dict, Optional, Tuple, Union
import torch
from torch import nn
from e3nn import o3

from torch_geometric.nn.pool import radius_graph
from torch_scatter import scatter
import torch_geometric
import math

from equiformer.nets.graph_attention_transformer import (
    LinearRS,
    SeparableFCTP,
    Vec2AttnHeads,
    AttnHeads2Vec,
    Activation,
    SmoothLeakyReLU,
    EquivariantDropout,
    GraphDropPath,
    FeedForwardNetwork,
    GaussianRadialBasisLayer,
    EdgeDegreeEmbeddingNetwork,
    sort_irreps_even_first,
    get_mul_0,
    get_norm_layer
)

from .utils import print_tensor_info

import logging
logger = logging.getLogger(__name__)


class AtomEmbeddingLayer(nn.Module):
    """Embeds categorical and continuous atomic features into irreps and raw encodings.
    
    Args:
        categorical_features: List[(num_categories, embedding_dim)] for each categorical input
        continuous_features: Optional[List[int]] dimensions of continuous features 
        irreps_out: o3.Irreps string for output irreps
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - node_feats: [num_atoms, irreps_out.dim] tensor of irrep embeddings  
            - node_attr: [num_atoms, sum(cat_sizes)] one-hot node attributes
    """
    def __init__(
        self,
        categorical_features: List[tuple[int, int]], 
        continuous_features: Optional[List[int]] = None,
        irreps_out: o3.Irreps = o3.Irreps.spherical_harmonics(2)
    ):
        super().__init__()
        
        self.num_categories = [num_cats for num_cats, _ in categorical_features]
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, dim)
            for num_cats, dim in categorical_features
        ])
        
        self.continuous_dim = sum(continuous_features) if continuous_features else 0
        total_embedded = sum(dim for _, dim in categorical_features) + self.continuous_dim
        
        self.project = LinearRS(
            irreps_in=o3.Irreps(f"{total_embedded}x0e"),
            irreps_out=irreps_out,
            bias=True
        )

    def forward(
        self,
        categorical_feats: torch.Tensor,
        continuous_feats: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            categorical_feats: [num_atoms, num_categorical] integer tensor
            continuous_feats: Optional[num_atoms, sum(continuous_features)] tensor
            
        Returns:
            node_feats: [num_atoms, irreps_out.dim] tensor of irrep embeddings
            node_attr: [num_atoms, sum(cat_sizes)] one-hot node attributes
        """
        # Split and embed categorical features
        cat_feats = [
            embed(categorical_feats[:,i]) 
            for i, embed in enumerate(self.categorical_embeddings)
        ]
        
        # Generate one-hot encodings 
        one_hot = [
            nn.functional.one_hot(categorical_feats[:,i], num_classes=num_cats)
            for i, num_cats in enumerate(self.num_categories)
        ]
        
        # Combine features
        if continuous_feats is not None:
            embedded = torch.cat([*cat_feats, continuous_feats], dim=-1)
            node_attr = torch.cat([*one_hot, continuous_feats], dim=-1) 
        else:
            embedded = torch.cat(cat_feats, dim=-1)
            node_attr = torch.cat(one_hot, dim=-1)
            
        node_feats = self.project(embedded)
        return node_feats, node_attr.float()
    

class GraphAttentionLayer(torch.nn.Module):
    """E3 equivariant graph attention layer with nonlinear edge messages.

    Original implementation disregards node attributes in the attention mechanism,
    here they are cated to the edge features (radial distance embeddings) for 
    determining tensor product weights, which are used to compute vectors that are projected
    into values and attention scores
    
    Args:
        irreps_node_feats: Input node feature irreps
        irreps_node_attr: Node attribute irreps (invariant scalars)  
        irreps_edge_input: Edge input feature irreps (invariant scalars)
        irreps_edge_attr: Edge geometric irreps (spherical harmonics)
        irreps_node_output: Output node feature irreps
        fc_neurons: Hidden layer sizes for radial networks
        irreps_head: Feature irreps per attention head
        num_heads: Number of attention heads
        irreps_pre_attn: Optional irreps before attention
        alpha_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        rescale_degree: Scale output by node degree
        output_attentions: Whether to output attention weights
    """
    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_input: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_node_output: o3.Irreps,
        fc_neurons: List[int],
        irreps_head: o3.Irreps,
        num_heads: int,
        irreps_pre_attn: Optional[o3.Irreps] = None,
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        rescale_degree: bool = False,
        output_attentions: bool = False
    ):
        super().__init__()
        
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_input = o3.Irreps(irreps_edge_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr) 
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_feats if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.output_attentions = output_attentions

        # Linear projections for node features
        self.message_src_proj = LinearRS(self.irreps_node_feats, self.irreps_pre_attn, bias=True)
        self.message_dst_proj = LinearRS(self.irreps_node_feats, self.irreps_pre_attn, bias=False)

        # Set up attention mechanism
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads = sort_irreps_even_first(irreps_attn_heads)[0].simplify()
        
        mul_alpha = get_mul_0(irreps_attn_heads)  # Number of scalar features
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps(f"{mul_alpha}x0e")

        # Edge message network
        incoming_dims = 2 * self.irreps_node_attr.dim + self.irreps_edge_input.dim
        fc_neurons = [incoming_dims] + fc_neurons 
        self.edge_message_net = SeparableFCTP(
            self.irreps_pre_attn,
            self.irreps_edge_attr, 
            self.irreps_pre_attn,
            fc_neurons,
            use_activation=True,
            norm_layer=None,
            internal_weights=False
        )

        # Networks for attention mechanism
        self.attention_score_net = LinearRS(self.edge_message_net.dtp.irreps_out, irreps_alpha)
        self.value_net = SeparableFCTP(
            self.irreps_pre_attn,
            self.irreps_edge_attr,
            irreps_attn_heads,
            fc_neurons=None,
            use_activation=False, 
            norm_layer=None,
            internal_weights=True
        )

        # Multi-head processing
        self.score_to_heads = Vec2AttnHeads(o3.Irreps(f"{mul_alpha_head}x0e"), num_heads)
        self.value_to_heads = Vec2AttnHeads(self.irreps_head, num_heads)
        self.score_activation = Activation(o3.Irreps(f"{mul_alpha_head}x0e"), [SmoothLeakyReLU(0.2)])
        self.heads_to_vec = AttnHeads2Vec(irreps_head)
        
        # Learned attention parameters
        attention_dot = torch.empty(1, num_heads, mul_alpha_head)
        torch_geometric.nn.inits.glorot(attention_dot)
        self.attention_dot = torch.nn.Parameter(
            attention_dot.transpose(0, -1).transpose(1, -1).contiguous()
            .transpose(1, -1).transpose(0, -1)
        )
        # self.attention_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        # torch_geometric.nn.inits.glorot(self.attention_dot)

        # Dropout modules
        self.attention_dropout = None if alpha_drop == 0.0 else torch.nn.Dropout(alpha_drop)
        self.output_projection = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_dropout = None if proj_drop == 0.0 else EquivariantDropout(
            self.irreps_node_output, proj_drop)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attr: torch.Tensor, 
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_embedding: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            node_feats: [num_nodes, irreps_node_feats.dim] Node features 
            node_attr: [num_nodes, irreps_node_attr.dim] Node attributes
            edge_src: [num_edges] Edge source indices
            edge_dst: [num_edges] Edge destination indices  
            edge_attr: [num_edges, irreps_edge_attr.dim] Edge attributes
            edge_embedding: [num_edges, num_basis] Edge embeddings
            batch_idx: Optional [num_nodes] batch indices
            
        Returns:
            node_feats: [num_nodes, irreps_node_output.dim] Updated features
            attention_weights: Optional attention scores if output_attentions=True
        """
        # Process node features into messages
        message_src = self.message_src_proj(node_feats)
        message_dst = self.message_dst_proj(node_feats)
        message = message_src[edge_src] + message_dst[edge_dst]

        # Combine edge embeddings with source/dest node attributes 
        src_attr = node_attr[edge_src]
        dst_attr = node_attr[edge_dst] 
        weight_feats = torch.cat([edge_embedding, src_attr, dst_attr], dim=-1)

        # Compute messages using weights from combined attributes
        weight = self.edge_message_net.dtp_rad(weight_feats) 
        message = self.edge_message_net.dtp(message, edge_attr, weight)

        # Extract attention scores and values
        scores = self.attention_score_net(message)
        scores = self.score_to_heads(scores)
        
        value = self.edge_message_net.lin(message) 
        value = self.edge_message_net.gate(value)
        value = self.value_net(value, edge_attr=edge_attr, edge_scalars=weight_feats)
        value = self.value_to_heads(value)

        # Compute attention weights
        # print_tensor_info("attention_dot", self.attention_dot)
        # print_tensor_info("scores-pre", scores) 
        scores = self.score_activation(scores)
        scores = torch.einsum('bik,aik->bi', scores, self.attention_dot)
        # print_tensor_info("scores-post", scores)
        scores = torch_geometric.utils.softmax(scores, edge_dst)
        scores = scores.unsqueeze(-1)

        if self.attention_dropout:
            scores = self.attention_dropout(scores)

        # Apply attention and aggregate
        out = value * scores
        out = scatter(out, edge_dst, dim=0, dim_size=node_feats.shape[0])
        out = self.heads_to_vec(out)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(
                edge_dst, num_nodes=node_feats.shape[0], dtype=node_feats.dtype)
            out = out * degree.view(-1, 1)

        # Final projection
        out = self.output_projection(out)
        if self.proj_dropout:
            out = self.proj_dropout(out)

        if not self.output_attentions:
            return out
        else:
            return out, scores


class TransformerBlockLayer(torch.nn.Module):
    """E3-equivariant transformer block with attention and feed-forward networks.

    First, node features are layer normed.
    Then, a graph attention update computes new node features.
    A skip connection from before attention is added to the output.
    Finally, a feed-forward network is applied to the output.
    Ouputs are layer normed.
    A skip connection from before the feed-forward network is added to the output.

    E3-equivariant transformer block with attention and feed-forward networks.
    
    Args:
        irreps_node_feats: Input node feature irreps
        irreps_node_attr: Node attribute irreps (invariant scalars)  
        irreps_edge_attr: Edge geometric irreps (spherical harmonics)
        irreps_edge_features: Edge radial features irreps
        irreps_node_output: Output node feature irreps
        irreps_mid: Optional intermediate irreps, defaults to output
        irreps_head: Feature irreps per attention head, defaults to output
        fc_neurons: Hidden layer sizes for attention messages
        num_heads: Number of attention heads
        rescale_degree: Scale output by node degree
        alpha_drop: Attention dropout rate
        proj_drop: Projection dropout rate
        drop_path_rate: Skip connection dropout rate
        norm_layer: Layer normalization type
        output_attentions: Whether to output attention weights
    """
    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_edge_features: o3.Irreps,
        irreps_node_output: o3.Irreps,
        irreps_mid: Optional[o3.Irreps]=None,
        irreps_head: Optional[o3.Irreps]=None,
        fc_neurons: List[int]=[32, 32],
        num_heads: int=4,
        rescale_degree: bool=False,
        alpha_drop: float=0.0,
        proj_drop: float=0.0,
        drop_path_rate: float=0.0,
        norm_layer: Optional[str]='layer',
        output_attentions: bool=False
    ):
        super().__init__()
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_edge_features = o3.Irreps(irreps_edge_features)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_mid = self.irreps_node_output if irreps_mid is None else o3.Irreps(irreps_mid)
        self.irreps_head = self.irreps_node_output if irreps_head is None else o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.drop_path_rate = drop_path_rate
        self.output_attentions = output_attentions

        # Layer norm
        self.norm_layer1 = get_norm_layer(norm_layer)(self.irreps_node_feats)
        self.norm_layer2 = get_norm_layer(norm_layer)(self.irreps_node_output)

        # graph attention
        self.graph_attention = GraphAttentionLayer(
            irreps_node_feats=self.irreps_node_feats,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_edge_input=self.irreps_edge_features,
            irreps_node_output=self.irreps_node_feats,
            irreps_head=self.irreps_head,
            fc_neurons=fc_neurons,
            num_heads=self.num_heads,
            alpha_drop=self.alpha_drop,
            proj_drop=self.proj_drop,
            rescale_degree=self.rescale_degree,
            output_attentions=self.output_attentions
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # output projection after attention
        self.head = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_output,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mid,
            proj_drop=self.proj_drop
        )

        # skip attention mechanism
        if self.irreps_node_feats == self.irreps_node_output:
            self.skip_connection = None
        else:
            self.skip_connection = FeedForwardNetwork(
                irreps_node_feats=self.irreps_node_feats,
                irreps_node_attr=self.irreps_node_attr,
                irreps_node_output=self.irreps_node_output,
                irreps_mlp_mid=self.irreps_mid,
            )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attr: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_embedding: torch.Tensor,
        batch_idx: Optional[torch.Tensor]=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            node_feats: [num_nodes, irreps_node_feats.dim] Node features 
            node_attr: [num_nodes, irreps_node_attr.dim] Node attributes
            edge_src: [num_edges] Edge source indices
            edge_dst: [num_edges] Edge destination indices
            edge_attr: [num_edges, irreps_edge_attr.dim] Edge attributes
            edge_embedding: [num_edges, num_basis] Edge embeddings
            batch_idx: Optional [num_nodes] batch indices
            
        Returns:
            node_feats: [num_nodes, irreps_node_output.dim] Updated features
            attention_weights: Optional attention scores if output_attentions=True
        """
        # Layer norm
        node_feats_input = node_feats
        node_feats = self.norm_layer1(node_feats, batch=batch_idx)

        # Attention mechanism
        node_feats_hidden = self.graph_attention(
            node_feats,
            node_attr,
            edge_src,
            edge_dst,
            edge_attr,
            edge_embedding,
            batch_idx
        )
        if self.output_attentions:
            node_feats_hidden, attentions = node_feats_hidden

        # drop path
        if self.drop_path:
            node_feats_hidden = self.drop_path(node_feats_hidden, batch_idx)

        # skip connection on attention
        if self.skip_connection is None:
            node_feats_hidden = node_feats_hidden + node_feats_input
        else:
            node_skip = self.skip_connection(
                node_feats_input,
                node_attr,
                edge_src,
                edge_dst,
                edge_attr,
                edge_embedding,
                batch_idx
            )
            node_feats_hidden = node_feats_hidden + node_skip

        # Store for second skip connection
        node_feats_output = node_feats_hidden

        # output projection
        node_feats_hidden = self.norm_layer2(node_feats_hidden, batch=batch_idx)
        node_feats_hidden = self.head(
            node_feats_hidden,
            node_attr,
        )

        # skip connection on output
        node_feats_output = node_feats_output + node_feats_hidden
        
        if self.output_attentions:
            return node_feats_output, attentions
        else:
            return node_feats_output
    

class MetalSiteFoundationalBackbone(torch.nn.Module):
    """E3-equivariant transformer model for learning protein metal site representations.

    Processes protein structure graphs using SO(3)-equivariant attention mechanisms. Takes atomic coordinates and identities as input 
    and produces equivariant node-level features that can be used for downstream tasks.

    Args:
        irreps_node_feats: Irreps for initial node embeddings. Default: 128x0e+64x1o+32x2e
        irreps_sh: Irreps for spherical harmonics. Default: up to l=3
        irreps_node_output: Irreps for final features. Default: same as node_embed
        atom_vocab_size: Number of atom types. Default: 14
        atom_type_vocab_size: Number of record types. Default: 3
        atom_embed_dim: Dimension for atomic feature embedding. Default: 16
        max_radius: Maximum radius (Å) for edges. Default: 6.0
        max_neighbors: Maximum number of neighbors. Default: 12
        num_basis: Number of radial basis functions. Default: 32 
        fc_neurons: Hidden layer sizes for radial networks
        irreps_head: Feature irreps per attention head
        num_heads: Number of attention heads. Default: 4
        num_layers: Number of transformer blocks. Default: 2
        alpha_drop: Attention dropout rate. Default: 0.0
        proj_drop: Projection dropout rate. Default: 0.0
        drop_path_rate: Skip connection dropout rate. Default: 0.0
        avg_num_neighbors: Expected neighbors for degree embedding. Default: 12
        output_attentions: Return attention weights
        output_hidden_states: Return all hidden states
    """
    def __init__(
        self,
        irreps_node_feats: o3.Irreps = o3.Irreps('128x0e+64x1o+32x2e'),
        irreps_sh: o3.Irreps = o3.Irreps.spherical_harmonics(3),
        irreps_node_output: o3.Irreps = o3.Irreps('128x0e+64x1o+32x2e'),
        atom_vocab_size: int = 14,
        atom_type_vocab_size: int = 3,
        atom_embed_dim: int = 16,
        max_radius: float = 6.0,
        max_neighbors: int = 12,
        num_basis: int = 32,
        fc_neurons: List[int] = [32, 32],
        irreps_head: o3.Irreps = o3.Irreps('32x0e+16x1o+8x2e'),
        num_heads: int = 4,
        num_layers: int = 2,
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        avg_num_neighbors: int = 12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()

        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.atom_vocab_size = atom_vocab_size
        self.atom_type_vocab_size = atom_type_vocab_size
        self.atom_embed_dim = atom_embed_dim
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.fc_neurons = fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.max_neighbors = max_neighbors

        oh_size = atom_vocab_size + atom_type_vocab_size

        self.irreps_node_attr = o3.Irreps(f"{oh_size}x0e")
        self.irreps_edge_features = o3.Irreps(f"{num_basis}x0e")
        self.irreps_edge_attr = self.irreps_sh

        # Embed atoms
        self.atom_embedder = AtomEmbeddingLayer(
            categorical_features=[
                (self.atom_vocab_size, self.atom_embed_dim),
                (self.atom_type_vocab_size, self.atom_embed_dim)
            ],
            continuous_features=None,
            irreps_out=self.irreps_node_feats
        )

        # Edge features
        self.radial_basis = GaussianRadialBasisLayer(
            self.num_basis, self.max_radius
        )

        # Initial geometric embedding
        self.edge_degree_embedding = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_feats,
            self.irreps_sh,
            fc_neurons=[self.num_basis] + self.fc_neurons,
            avg_aggregate_num=avg_num_neighbors
        )

        # Transformer blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(num_layers):
            irreps_out = irreps_node_output if i == num_layers - 1 else irreps_node_feats
            
            block = TransformerBlockLayer(
                irreps_node_feats=self.irreps_node_feats,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_edge_features=self.irreps_edge_features,
                irreps_node_output=irreps_out,
                num_heads=self.num_heads,
                drop_path_rate=drop_path_rate,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
                output_attentions=output_attentions
            )
            self.blocks.append(block)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        atom_tokens: torch.Tensor,  
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone.
        
        Args:
            atom_tokens: [num_nodes, 2] Node categorical features (atoms and types)
            pos: [num_nodes, 3] Node coordinates
            batch_idx: [num_nodes] Batch assignments
            
        Returns:
            Dict containing:
                node_feats: [num_nodes, irreps_node_output.dim] Output features
                node_attr: [num_nodes, irreps_node_attr.dim] Node attributes
                edge_attr: Optional[num_edges, irreps_edge_attr.dim] Edge attributes
                hidden_states: Optional List of intermediate features
                attentions: Optional List of attention weights
        """
        # Get edges
        edge_src, edge_dst = radius_graph(
            pos, r=self.max_radius, batch=batch_idx, loop=False, max_num_neighbors=self.max_neighbors
        )
        logger.debug(f"Edges in batch: {edge_src.shape[0]}")
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        
        # Edge features
        edge_attr = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization='component'
        )
        edge_lengths = edge_vec.norm(dim=-1)
        edge_embedding = self.radial_basis(edge_lengths)

        # Node features
        node_feats, node_attr = self.atom_embedder(atom_tokens)

        # Incorporate geometry
        edge_degree_embedding = self.edge_degree_embedding(
            node_feats,
            edge_attr,
            edge_embedding,
            edge_src,
            edge_dst,
            batch_idx
        )
        node_feats = node_feats + edge_degree_embedding

        # Store intermediate states
        hidden_states = []
        attentions = []

        # Forward through blocks
        for block in self.blocks:
            if self.output_attentions:
                node_feats, block_attentions = block(
                    node_feats, node_attr,
                    edge_src, edge_dst,
                    edge_attr, edge_embedding,
                    batch_idx
                )
                attentions.append(block_attentions)
            else:
                node_feats = block(
                    node_feats, node_attr,
                    edge_src, edge_dst,
                    edge_attr, edge_embedding,
                    batch_idx
                )
            
            if self.output_hidden_states:
                hidden_states.append(node_feats)

        outputs = {
            'node_feats': node_feats,
            'node_attr': node_attr,
            'edge_attr': edge_attr,
            'edge_embedding': edge_embedding
        }

        if self.output_hidden_states:
            outputs['hidden_states'] = hidden_states
            
        if self.output_attentions:
            outputs['attentions'] = attentions

        return outputs
    

class MetalSiteNodeHeadLayer(torch.nn.Module):
    """Output logits over vocab and a vector.
    
    Note that there is no scaling factor applied here, thus this module should be trained
    on normed/scaled quantities.

    Args:
        irreps_node_feats: Input node feature irreps
        irreps_node_attr: Node attribute irreps
        proj_drop: Dropout rate for output projection
        atom_vocab_size: Number of atom types
        atom_type_vocab_size: Number of record types
        
    Returns:
        atom_logits: [num_nodes, atom_vocab_size] Atom type predictions
        type_logits: [num_nodes, record_vocab_size] Record type predictions  
        coordinates: [num_nodes, 3] Coordinate adjustments
    """
    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        proj_drop: float = 0.0,
        atom_vocab_size: int = 14,
        atom_type_vocab_size: int = 3,
    ):
        super().__init__()
        
            
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.proj_drop = proj_drop
        self.atom_vocab_size = atom_vocab_size
        self.atom_type_vocab_size = atom_type_vocab_size

        self.n_tokens = self.atom_vocab_size + self.atom_type_vocab_size
        self.irreps_output = o3.Irreps(f"{self.n_tokens}x0e+1x1o")

        # Layer norm & FFN 
        self.norm = get_norm_layer('layer')(self.irreps_node_feats)
        self.head = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_feats,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_output,
            proj_drop=self.proj_drop
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attr: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through prediction head.
        
        Args:
            node_feats: [num_nodes, irreps_node_feats.dim] Node features
            node_attr: [num_nodes, irreps_node_attr.dim] Node attributes
            
        Returns:
            atom_logits: [num_nodes, atom_vocab_size] Atom type predictions
            type_logits: [num_nodes, record_vocab_size] Record type predictions
            coordinates: [num_nodes, 3] Coordinate adjustments
        """
        node_feats = self.norm(node_feats)
        outputs = self.head(node_feats, node_attr)
        
        # Split outputs into logits and coordinates
        atom_logits = outputs[:, :self.atom_vocab_size]
        type_logits = outputs[:, self.atom_vocab_size:self.n_tokens]
        assert type_logits.shape[1] == self.atom_type_vocab_size
        coordinates = outputs[:, self.n_tokens:]
        
        return atom_logits, type_logits, coordinates
