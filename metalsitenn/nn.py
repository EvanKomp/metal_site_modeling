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
from typing import List, Dict, Optional
import torch
from torch import nn
from e3nn import o3

from torch_cluster import radius_graph
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

from metalsitenn.atom_vocabulary import AtomTokenizer

class AtomEmbedding(nn.Module):
    """Embeds categorical and continuous atomic features into irreps and raw encodings.
    
    Args:
        categorical_features: List of (num_categories, embedding_dim) tuples
        continuous_features: List of feature dimensions 
        irreps_out: Output irreps string
        
    Returns:
        Tuple of (embeddings: irreps_out, raw: a one hot encoding)
    """
    def __init__(
        self,
        categorical_features: List[tuple[int, int]], 
        continuous_features: List[int]=None,
        irreps_out: str = o3.Irreps.spherical_harmonics(2)
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
        categorical_features: torch.Tensor,
        continuous_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            categorical_features: [num_atoms, num_categorical] integer tensor
            continuous_features: [num_atoms, sum(continuous_features)] tensor
        """
        # Split and embed categorical features
        cat_feats = [
            embed(categorical_features[:,i]) 
            for i, embed in enumerate(self.categorical_embeddings)
        ]
        
        # Generate one-hot encodings 
        one_hot = [
            nn.functional.one_hot(categorical_features[:,i], num_classes=num_cats)
            for i, num_cats in enumerate(self.num_categories)
        ]
        
        # Combine features
        if continuous_features is not None:
            embedded = torch.cat([*cat_feats, continuous_features], dim=-1)
            raw = torch.cat([*one_hot, continuous_features], dim=-1) 
        else:
            embedded = torch.cat(cat_feats, dim=-1)
            raw = torch.cat(one_hot, dim=-1)
            
        return self.project(embedded), raw.float()
    

class GraphAttention(torch.nn.Module):
    """E3 equivariant graph attention layer with nonlinear edge messages.

    Original implementation discards node attributes. Here they are combined with edge distance embedding to
    help determine tensor proruct weights.

    Node features are used to compute messages, queries, and keys. Node attributes and 
    edge embeddings together determine the weights for tensor products. This ensures
    node attributes only influence the strength of interactions while maintaining
    equivariance through the node features.

    Args:
        irreps_node_input: Input node feature irreps
        irreps_node_attr: Node attribute irreps (invariant scalars)  
        irreps_edge_attr: Edge geometric irreps (spherical harmonics)
        irreps_node_output: Output node feature irreps
        fc_neurons: Hidden layer sizes for radial networks
        irreps_head: Feature irreps per attention head
        num_heads: Number of attention heads
        irreps_pre_attn: Optional irreps before attention
        alpha_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection,
        rescale_degree: Scale output
    """
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
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
    ):
        super().__init__()
        
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_input = o3.Irreps(irreps_edge_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr) 
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree

        # Linear projections for node features
        self.message_src_proj = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.message_dst_proj = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)

        # Set up attention mechanism
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads = sort_irreps_even_first(irreps_attn_heads)[0].simplify()
        
        mul_alpha = get_mul_0(irreps_attn_heads)  # Number of scalar features
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps(f"{mul_alpha}x0e")
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        # Edge message network - note fc_neurons processes concatenated edge_embedding + node_attr
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
        self.attention_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.attention_dot)

        # Dropout modules
        self.attention_dropout = None if alpha_drop == 0.0 else torch.nn.Dropout(alpha_drop)
        self.output_projection = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_dropout = None if proj_drop == 0.0 else EquivariantDropout(
            self.irreps_node_output, proj_drop)

    def forward(
        self,
        node_input: torch.Tensor,
        node_attr: torch.Tensor, 
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_embedding: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            node_input: Node features of shape [num_nodes, irreps_node_input.dim]
            node_attr: Node attributes of shape [num_nodes, irreps_node_attr.dim] 
            edge_src: Edge source indices of shape [num_edges]
            edge_dst: Edge destination indices of shape [num_edges]
            edge_attr: Edge attributes (geometric) of shape [num_edges, irreps_edge_attr.dim]
            edge_embedding: Edge radial features of shape [num_edges, num_basis]
            batch: Optional batch indices of shape [num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, irreps_node_output.dim]
        """
        # Process node features into messages
        message_src = self.message_src_proj(node_input)
        message_dst = self.message_dst_proj(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        # Combine edge embeddings with source/dest node attributes for tensor product weights
        src_attr = node_attr[edge_src]
        dst_attr = node_attr[edge_dst] 
        weight_features = torch.cat([edge_embedding, src_attr, dst_attr], dim=-1)

        # Compute messages using weights determined by combined attributes
        weight = self.edge_message_net.dtp_rad(weight_features) 
        message = self.edge_message_net.dtp(message, edge_attr, weight)

        # Extract attention scores and values
        scores = self.attention_score_net(message)
        scores = self.score_to_heads(scores)
        
        value = self.edge_message_net.lin(message) 
        value = self.edge_message_net.gate(value)
        value = self.value_net(value, edge_attr=edge_attr, edge_scalars=weight_features)
        value = self.value_to_heads(value)

        # Compute attention weights
        scores = self.score_activation(scores)
        scores = torch.einsum('bik,aik->bi', scores, self.attention_dot)
        scores = torch_geometric.utils.softmax(scores, edge_dst)
        scores = scores.unsqueeze(-1)

        if self.attention_dropout:
            scores = self.attention_dropout(scores)

        # Apply attention and aggregate
        out = value * scores
        out = scatter(out, edge_dst, dim=0, dim_size=node_input.shape[0])
        out = self.heads_to_vec(out)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(
                edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype)
            out = out * degree.view(-1, 1)

        # Final projection
        out = self.output_projection(out)
        if self.proj_dropout:
            out = self.proj_dropout(out)

        return out


class TransformerBlock(torch.nn.Module):
    """E3-equivariant transformer block with attention and feed-forward networks.

    First, node features are layer normed.
    Then, a graph attention update computes new node features.
    A skip connection from before attention is added to the output.
    Finally, a feed-forward network is applied to the output.
    Ouputs are layer normed.
    A skip connection from before the feed-forward network is added to the output.

    Args:
        irreps_node_input: Input node feature irreps
        irreps_node_attr: Node attribute irreps (invariant scalars)  
        irreps_edge_attr: Edge geometric irreps (spherical harmonics)
        irreps_edge_features: Edge radial features irreps (invariant scalars)
        irreps_node_output: Output node feature irreps
        irreps_mid: Intermediate node feature irreps, defaults to output
        irreps_head: Feature irreps per attention head, defaults to output
            recommended to have this be of the same order as the output but smaller by a factor
            of the number of heads
        fc_neurons: Hidden layer sizes for radial networks
        num_heads: Number of attention heads
        rescale_degree: Scale output
        alpha_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection,
        drop_path_rate: Dropout rate for skip connections
        norm_layer: Layer normalization
    """

    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_edge_features: o3.Irreps,
        irreps_node_output: o3.Irreps,
        irreps_mid: o3.Irreps=None,
        irreps_head: o3.Irreps=None,
        fc_neurons: List[int]=[32, 32], # for sending messages
        num_heads: int=4,
        rescale_degree: bool=False,
        alpha_drop: float=0.0,
        proj_drop: float=0.0,
        drop_path_rate: float=0.0,
        norm_layer: Optional[str]='layer',
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
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

        # Layer norm
        self.norm_layer1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.norm_layer2 = get_norm_layer(norm_layer)(self.irreps_node_output)

        # graph attention
        self.graph_attention = GraphAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_edge_input=self.irreps_edge_features,
            irreps_node_output=self.irreps_node_input, # output is same as input, so is internal state since it was not passed
            irreps_head=self.irreps_head,
            fc_neurons=fc_neurons,
            num_heads=self.num_heads,
            alpha_drop=self.alpha_drop,
            proj_drop=self.proj_drop,
            rescale_degree=self.rescale_degree
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
        if self.irreps_node_input == self.irreps_node_output:
            self.skip_connection = None
        else:
            self.skip_connection = FeedForwardNetwork(
                irreps_node_input=self.irreps_node_input,
                irreps_node_attr=self.irreps_node_attr,
                irreps_node_output=self.irreps_node_output,
                irreps_mlp_mid=self.irreps_mid,
            )

        # ffn also skipped but proper irreps already met

    def forward(
        self,
        node_input: torch.Tensor,
        node_attr: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_embedding: torch.Tensor,
        batch: Optional[torch.Tensor]=None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            node_input: Node features of shape [num_nodes, irreps_node_input.dim]
            node_attr: Node attributes of shape [num_nodes, irreps_node_attr.dim] 
            edge_src: Edge source indices of shape [num_edges]
            edge_dst: Edge destination indices of shape [num_edges]
            edge_attr: Edge attributes (geometric) of shape [num_edges, irreps_edge_attr.dim]
            edge_embedding: Edge radial features of shape [num_edges, num_basis]
            batch: Optional batch indices of shape [num_nodes]
            
        Returns:
            Updated node features of shape [num_nodes, irreps_node_output.dim]
        """
        # Layer norm
        node_start = node_input
        node_input = self.norm_layer1(node_input, batch=batch)

        # Attention mechanism
        node_hidden_state = self.graph_attention(
            node_input,
            node_attr,
            edge_src,
            edge_dst,
            edge_attr,
            edge_embedding,
            batch
        )

        # drop path
        if self.drop_path:
            node_hidden_state = self.drop_path(node_hidden_state, batch)

        # skip connection on attention
        if self.skip_connection is None:
            node_hidden_state = node_hidden_state + node_start
        else:
            node_skip = self.skip_connection(
                node_start,
                node_attr,
                edge_src,
                edge_dst,
                edge_attr,
                edge_embedding,
                batch
            )
            node_hidden_state = node_hidden_state + node_skip

        # We will also send post attention to skip
        node_output = node_hidden_state

        # output projection
        node_hidden_state = self.norm_layer2(node_hidden_state, batch=batch)
        node_hidden_state = self.head(
            node_hidden_state,
            node_attr,
        )

        # skip connection on output
        node_output = node_output + node_hidden_state
        return node_output
    

class MetalSiteFoundationalModel(torch.nn.Module):
    """E3-equivariant transformer model for learning protein metal site representations.

    Processes protein structure graphs using SO(3)-equivariant attention mechanisms. Takes atomic coordinates and identities as input 
    and produces equivariant node-level features that can be used for downstream tasks.

    Args:
        irreps_node_embedding: Irreps for initial node embeddings after projection. Default: 128 scalars, 64 vectors, 32 l=2
        irreps_sh: Irreps for spherical harmonics edge features. Default: up to l=3
        irreps_output: Irreps for final node features. Default: same as node embedding
        tokenizer: Vocabulary for converting atomic info to indices
        atom_embedding_dims: Dimension for atomic feature embedding. Default: 16
        max_radius: Maximum radius in Angstroms for constructing edges. Default: 6.0
        number_basis: Number of radial basis functions. Default: 32 
        fc_neurons: Hidden layer sizes for radial networks. Default: [32, 32]
        irreps_head: Feature irreps per attention head. Default: 32 scalars, 16 vectors, 8 l=2
        num_head: Number of attention heads. Default: 4
        num_layers: Number of transformer blocks. Default: 2
        alpha_drop: Dropout rate for attention weights. Default: 0.0
        proj_drop: Dropout rate for projections. Default: 0.0
        out_drop: Dropout rate for outputs. Default: 0.0
        drop_path_rate: Dropout rate for skip connections. Default: 0.0
        avg_aggregate_num: Number of neighbors for edge degree embedding. Default: 12

    Returns:
        node_hidden_state: Node features with irreps_output symmetry
        node_attrs: Original one-hot node attributes

    Raises:
        ValueError: If tokenizer is not provided
    """
    def __init__(
        self,
        irreps_node_embedding: o3.Irreps=o3.Irreps('128x0e+64x1o+32x2e'),
        irreps_sh: o3.Irreps=o3.Irreps.spherical_harmonics(3),
        irreps_output: o3.Irreps=o3.Irreps('128x0e+64x1o+32x2e'),
        tokenizer: AtomTokenizer=None,
        atom_embedding_dims: int=16,
        max_radius: float=6.0,
        number_basis: int=32,
        fc_neurons: List[int]=[32, 32],
        irreps_head: o3.Irreps=o3.Irreps('32x0e+16x1o+8x2e'),
        num_head: int=4,
        num_layers: int=2,
        alpha_drop: float=0.0,
        proj_drop: float=0.0,
        out_drop: float=0.0,
        drop_path_rate: float=0.0,
        avg_aggregate_num: int=12
    ):
        super().__init__()

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_output = o3.Irreps(irreps_output)
        self.tokenizer = tokenizer
        self.atom_embedding_dims = atom_embedding_dims
        self.max_radius = max_radius
        self.number_basis = number_basis
        self.fc_neurons = fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_head
        self.num_layers = num_layers
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.avg_aggregate_num = avg_aggregate_num

        self.irreps_node_attr = o3.Irreps(f"{self.tokenizer.oh_size}x0e")
        self.irreps_edge_features = o3.Irreps(f"{number_basis}x0e")
        self.irreps_edge_attr = self.irreps_sh

        # embed the atoms
        if self.tokenizer is None:
            raise ValueError("Vocabulary must be provided for atom embedding")
        self.atom_embedder = AtomEmbedding(
            categorical_features=[
                (self.tokenizer.atom_vocab.vocab_size, self.atom_embedding_dims),
                (self.tokenizer.record_vocab.vocab_size, self.atom_embedding_dims)
            ],
            continuous_features=None,
            irreps_out=self.irreps_node_embedding
        )

        # radial basis features
        self.radial_basis = GaussianRadialBasisLayer(
            self.number_basis, self.max_radius
        )

        # incorporate geometry into initial embedding
        self.edge_deg_embedding = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding,
            self.irreps_sh,
            fc_neurons=[self.number_basis] + self.fc_neurons,
            avg_aggregate_num=self.avg_aggregate_num
        )

        # transformer blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                irreps_output = self.irreps_node_embedding
            else:
                irreps_output = self.irreps_output

            block = TransformerBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_edge_features=self.irreps_edge_features,
                irreps_node_output=irreps_output,
                num_heads=self.num_heads,
                drop_path_rate=self.drop_path_rate,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop
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
            self, atom_identifiers, positions, batch_indices, **kwargs):
        
        # get edges
        edge_src, edge_dst = radius_graph(
            positions, r=self.max_radius, batch=batch_indices, loop=False
        )
        edge_vec = positions.index_select(0, edge_src) - positions.index_select(0, edge_dst)
        edge_attributes = o3.spherical_harmonics(l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component')
        edge_distances = edge_vec.norm(dim=-1)
        edge_features = self.radial_basis(edge_distances)

        # and nodes
        node_features, node_attrs = self.atom_embedder(atom_identifiers)

        # incorporate geometry into nodes
        edge_degree_embedding = self.edge_deg_embedding(
            node_features, edge_attributes, edge_features, edge_src, edge_dst, batch_indices
        )
        node_hidden_state = node_features + edge_degree_embedding

        # run through transformer blocks
        for block in self.blocks:
            node_hidden_state = block(
                node_hidden_state, node_attrs, edge_src, edge_dst, edge_attributes, edge_features, batch_indices
            )

        return node_hidden_state, node_attrs
    

class MetalSiteNodeHead(torch.nn.Module):
    """Output logits over vocab and a vector.
    
    Note that there is no scaling factor applied here, thus this module should be trained
    on normed/scalled quantities.

    Args
        irreps_node_input: Input node feature irreps
        irreps_node_attrs: Node attribute irreps
        proj_drop: Dropout rate for output projection
        tokenizer: AtomTokenizer
    """
    def __init__(
        self,
        irreps_node_input: o3.Irreps=o3.Irreps('128x0e+64x1o+32x2e'),
        irreps_node_attrs: o3.Irreps=o3.Irreps('1x0e'),
        proj_drop: float=0.0,
        tokenizer: AtomTokenizer=None
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)
        self.proj_drop = proj_drop
        if tokenizer is None:
            raise ValueError("Vocabulary must be provided for atom embedding")
        self.tokenizer = tokenizer

        self.n_tokens = self.tokenizer.oh_size
        self.irreps_output = o3.Irreps(f"{self.n_tokens}x0e+1x1o")

        # get layer norm
        self.norm_layer = get_norm_layer('layer')(self.irreps_node_input)

        # ffn
        self.head = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attrs,
            irreps_node_output=self.irreps_output,
            proj_drop=self.proj_drop
        )

    def forward(
        self,
        node_input: torch.Tensor,
        node_attrs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        node_input = self.norm_layer(node_input)
        node_output = self.head(node_input, node_attrs)
        
        # extract scalaras and vectors
        atom_scalars = node_output[:, :self.tokenizer.atom_vocab.vocab_size]
        atom_type_scalars = node_output[:, self.tokenizer.atom_vocab.vocab_size:self.n_tokens]
        vectors = node_output[:, self.n_tokens:]
        return atom_scalars, atom_type_scalars, vectors


class MetalSiteNodeModel(torch.nn.Module):
    """Combines foundational model and head for a full model that can be trained simultaneously for atom demasking and
    position denoising."""
    
    def __init__(
        self,
        node_model: MetalSiteFoundationalModel,
        node_head: MetalSiteNodeHead
    ):
        super().__init__()
        self.node_model = node_model
        self.node_head = node_head

    def forward(
        self,
        atom_identifiers, positions, batch_indices, **kwargs
    ):
        node_hidden_state, node_attrs = self.node_model(
            atom_identifiers, positions, batch_indices
        )
        atom_scalars, atom_type_scalars, vectors = self.node_head(node_hidden_state, node_attrs)
        return atom_scalars, atom_type_scalars, vectors
    

def get_irreps(l: int, scale: int, decay: float, num_heads: int = None) -> tuple[o3.Irreps, o3.Irreps]:
    """Generate scalable irreps for network features and attention heads.
    
    Args:
        l: Maximum order of irreps
        scale: Base multiplicity for l=0 irreps
        decay: Factor to reduce multiplicity for each l (decay^l)
        num_heads: If provided, returns attention head irreps with multiplicities divided by num_heads
    
    Returns:
        hidden_irreps: Full irreps for hidden features
        head_irreps: Irreps for attention heads if num_heads provided, else None
    """
    irreps = []
    head_irreps = []
    
    for i in range(l + 1):
        # Calculate multiplicity with decay
        mult = int(scale * (decay ** i))
        if mult == 0:
            continue
            
        # Add even and odd irreps of order i
        irreps.extend([
            (mult, (i, 1)),  # even
            (mult, (i, -1))  # odd
        ])
        
        # Calculate head multiplicities if requested
        if num_heads is not None:
            head_mult = math.ceil(mult / num_heads)
            if head_mult > 0:
                head_irreps.extend([
                    (head_mult, (i, 1)),
                    (head_mult, (i, -1))
                ])
    
    hidden_irreps = o3.Irreps(irreps)
    head_irreps = o3.Irreps(head_irreps) if num_heads is not None else None
    
    return hidden_irreps, head_irreps