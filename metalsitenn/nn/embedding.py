# metalsitenn/nn/embedding.py
'''
* Author: Evan Komp
* Created: 8/13/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT


This module provides embedding classes for converting discrete molecular features
(atoms, bonds) into continuous representations and projecting them into SO3 format.
'''

import copy
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fairchem.core.models.equiformer_v2.radial_function import RadialFunction
from fairchem.core.models.equiformer_v2.so3 import (
    SO3_Embedding,
)


class NodeEmbedder(nn.Module):
    """
    Embed atom features, concat, then project to output dimension.
    
    Handles all atom-level features in the molecular graph.
    
    Args:
        feature_vocab_sizes: Dict mapping atom feature names to vocab sizes
        atom_features: List of atom feature names
        output_dim: Output dimension for concatenated atom features
        embedding_dim: Individual embedding dimension per feature
        use_bias: Whether to use bias in final projection layer
    """
    
    def __init__(
        self,
        feature_vocab_sizes: Dict[str, int],
        atom_features: List[str],
        output_dim: int = 64,
        embedding_dim: int = 32,
        use_bias: bool = True
    ):
        super().__init__()
        
        self.atom_features = atom_features
        
        # Separate embeddings for each atom feature
        self.embeddings = nn.ModuleDict()
        for feature in atom_features:
            vocab_size = feature_vocab_sizes[feature]
            self.embeddings[feature] = nn.Embedding(vocab_size, embedding_dim)
            
        # Project concatenated features to desired output dimension
        concat_dim = len(atom_features) * embedding_dim
        self.projection = nn.Linear(concat_dim, output_dim, bias=use_bias)
        
    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed and project atom features.
        
        Args:
            feature_dict: Dict of feature_name -> token_indices tensor
            
        Returns:
            Atom embeddings tensor of shape (num_atoms, output_dim)
        """
        # Embed each atom feature individually
        embedded_features = []
        for name in self.atom_features:
            tokens = feature_dict[name]
            embedded = self.embeddings[name](tokens.squeeze(-1))
            embedded_features.append(embedded)
            
        # Concatenate all atom features
        atom_embeds = torch.cat(embedded_features, dim=-1)
        
        # Project to output dimension
        atom_embeds = self.projection(atom_embeds)
        return atom_embeds


class EdgeEmbedder(nn.Module):
    """
    Embed bond features, concat, then project to output dimension.
    
    Handles all edge-level features in the molecular graph.
    
    Args:
        feature_vocab_sizes: Dict mapping bond feature names to vocab sizes
        bond_features: List of bond feature names
        output_dim: Output dimension for concatenated bond features
        embedding_dim: Individual embedding dimension per feature
        use_bias: Whether to use bias in final projection layer
    """
    
    def __init__(
        self,
        feature_vocab_sizes: Dict[str, int],
        bond_features: List[str],
        output_dim: int = 64,
        embedding_dim: int = 32,
        use_bias: bool = True
    ):
        super().__init__()
        
        self.bond_features = bond_features
        
        # Separate embeddings for each bond feature
        self.embeddings = nn.ModuleDict()
        for feature in bond_features:
            vocab_size = feature_vocab_sizes[feature]
            self.embeddings[feature] = nn.Embedding(vocab_size, embedding_dim)
            
        # Project concatenated features to desired output dimension
        concat_dim = len(bond_features) * embedding_dim
        self.projection = nn.Linear(concat_dim, output_dim, bias=use_bias)
        
    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed and project bond features.
        
        Args:
            feature_dict: Dict of feature_name -> token_indices tensor
            
        Returns:
            Bond embeddings tensor of shape (num_bonds, output_dim)
        """
        # Embed each bond feature individually
        embedded_features = []
        for name in self.bond_features:
            tokens = feature_dict[name]
            embedded = self.embeddings[name](tokens.squeeze(-1))
            embedded_features.append(embedded)
            
        # Concatenate all bond features
        bond_embeds = torch.cat(embedded_features, dim=-1)
        
        # Project to output dimension
        bond_embeds = self.projection(bond_embeds)
        return bond_embeds


class SO3ScalarEmbedder(nn.Module):
    """
    Converts pre-computed atom embeddings to SO3 embeddings by projecting them 
    to the l=0, m=0 coefficients across multiple resolutions.
    
    Args:
        lmax_list: List of maximum degrees (l) for each resolution
        sphere_channels: Number of spherical channels per resolution
    """
    
    def __init__(
        self,
        lmax_list: List[int],
        sphere_channels: int,
    ):
        super().__init__()
        
        self.lmax_list = lmax_list
        self.sphere_channels = sphere_channels
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * sphere_channels

    def forward(self, atom_embeddings: torch.Tensor) -> SO3_Embedding:
        """
        Convert atom embeddings to SO3 embeddings.
        
        Args:
            atom_embeddings: Input atom embeddings of shape (N, sphere_channels_all)
            
        Returns:
            SO3_Embedding: SO3 embedding with l=0, m=0 coefficients initialized
        """
        num_atoms = atom_embeddings.shape[0]
        if atom_embeddings.shape[1] != self.sphere_channels_all:
            raise ValueError(
                f"Expected atom_embeddings shape (N, {self.sphere_channels_all}), "
                f"but got {atom_embeddings.shape}"
            )
        
        # Initialize SO3 embedding
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            atom_embeddings.device,
            atom_embeddings.dtype,
        )
        
        # Fill in the l=0, m=0 coefficients for each resolution
        offset_res = 0  # Offset in SO3 embedding coefficient dimension
        offset_channels = 0  # Offset in projected embedding channels
        
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                # Single resolution case - use all projected channels
                x.embedding[:, offset_res, :] = atom_embeddings
            else:
                # Multi-resolution case - split channels across resolutions
                x.embedding[:, offset_res, :] = atom_embeddings[
                    :, offset_channels : offset_channels + self.sphere_channels
                ]
            
            # Update offsets for next resolution
            offset_channels += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)
        
        return x


class EdgeProjector(nn.Module):
    """
    Embed edges to output of target size.

    In equiformer, the radial basis distance is optionally combined with a src and dst node embedding,
    then projected with an MLP "radial_func" to a target size - the target size depends on the application.
    This class is meant to extract that functionality out and allow us to use all the extra node and edge features we
    have beyond just the distance and atomic identity.

    Args:
        radial_basis_size (int): Size of RBF expected
        feature_vocab_sizes (Dict[str, int]): Dictionary mapping feature names to vocab sizes
        use_edge_features (bool): Whether to use edge features
        bond_features (List[str]): List of bond feature names to use
        use_node_features (bool): Whether to use node features
        node_features (List[str]): List of node feature names to use
        output_dim (int): Output dimension for the edge embeddings
        embedding_dim (int): Embedding dimension for node and edge features from NodeEmbedder and EdgeEmbedder
        embedding_use_bias (bool): Whether to use bias in the embedding layers
        projector_hidden_layers (int): Number of hidden layers in the projector Radial func
        projector_output_size (int): Output size of the projector Radial func

    """
    
    def __init__(
        self,
        radial_basis_size: int,
        feature_vocab_sizes: Dict[str, int]={},
        use_edge_features: bool=True,
        bond_features: List[str]=['bond_order', 'is_in_ring', 'is_aromatic'],
        use_node_features: bool=True,
        node_features: List[str]=['element', 'charge', 'nhyd', 'hyb'],
        output_dim: int = 64,
        embedding_dim: int = 32,
        embedding_use_bias: bool = True,
        use_projector: bool = True,
        projector_hidden_layers: int = 1,
        projector_size: int = 64
    ):
        super().__init__()

        self.radial_basis_size = radial_basis_size
        self.feature_vocab_sizes = feature_vocab_sizes
        self.use_edge_features = use_edge_features
        self.bond_features = bond_features
        self.use_node_features = use_node_features
        self.node_features = node_features
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.use_bias = embedding_use_bias
        self.projector_hidden_layers = projector_hidden_layers
        self.projector_size = projector_size
        self.use_projector = use_projector

        # if we are using node features, create embedders
        if self.use_node_features:
            self.source_embedding = NodeEmbedder(
                feature_vocab_sizes=self.feature_vocab_sizes,
                atom_features=self.node_features,
                output_dim=embedding_dim,
                embedding_dim=embedding_dim,
                use_bias=embedding_use_bias
            )
            self.destination_embedding = NodeEmbedder(
                feature_vocab_sizes=self.feature_vocab_sizes,
                atom_features=self.node_features,
                output_dim=embedding_dim,
                embedding_dim=embedding_dim,
                use_bias=embedding_use_bias
            )
        else:
            self.source_embedding = None
            self.destination_embedding = None

        # if we are using edge features, create embedder
        if self.use_edge_features:
            self.edge_embedding = EdgeEmbedder(
                feature_vocab_sizes=self.feature_vocab_sizes,
                bond_features=self.bond_features,
                output_dim=embedding_dim,
                embedding_dim=embedding_dim,
                use_bias=embedding_use_bias
            )
        else:
            self.edge_embedding = None

        # get the epected input size for the radial function
        input_size = radial_basis_size
        if self.use_edge_features:
            input_size += embedding_dim
        if self.use_node_features:
            input_size += 2 * embedding_dim

        self.input_size = input_size

        # radial function to project the input to the output dimension
        if self.use_projector:
            channels_list = [input_size] + [self.projector_size] * self.projector_hidden_layers + [self.output_dim]
            self.radial_func = RadialFunction(
                channels_list=channels_list,
            )

    def forward(
        self,
        R: torch.Tensor, # [E, radial_basis_size]
        edge_index: torch.Tensor, # [E,2]
        feature_dict: Dict[str, torch.Tensor]={},
    ):
        to_concat = []
        # radial basis distance
        to_concat.append(R)

        # edge features
        if self.use_edge_features:
            edge_features = self.edge_embedding(feature_dict)
            to_concat.append(edge_features)

        # node features
        if self.use_node_features:
            nodes_embedded = self.source_embedding(feature_dict)

            # Extract the source and destination node embeddings
            src_embeddings = nodes_embedded[edge_index[:, 0]]
            dst_embeddings = nodes_embedded[edge_index[:, 1]]
            # Concatenate source and destination node embeddings
            to_concat.append(src_embeddings)
            to_concat.append(dst_embeddings)

        # concatenate all features
        concatenated = torch.cat(to_concat, dim=-1)
        if self.use_projector:
            # pass through radial function
            output = self.radial_func(concatenated)

            return output
        else:
            return concatenated


class EdgeDegreeEmbedding(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated

        DEPRECATED, using EdgeProjector instead
        # max_num_elements (int):     Maximum number of atomic numbers
        # edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        # use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        radial_basis_size (int):     Number of radial basis functions expected
        feature_vocab_sizes (list:int): List of sizes of feature vocabularies
        use_edge_features (bool):    Whether to use edge features
        bond_features (list:str): List of bond feature names to use if using any
        use_node_features (bool): Whether to use node features
        node_features (list:str): List of node feature names to use if using any
        embedding_dim (int):        Embedding dimension for node and edge features
        embedding_use_bias (bool):  Whether to use bias in the embedding layers
        projector_hidden_layers (int): Number of hidden layers in the projector Radial func
        projector_size (int):       Hidden layer size of the projector Radial func
        NOTE: Output size of radial func is determined by number of m0 coefficients available.

        rescale_factor (float):     Rescale the sum aggregation
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_rotation,
        mappingReduced,
        radial_basis_size: int,
        feature_vocab_sizes: Dict[str, int]={},
        use_edge_features: bool=True,
        bond_features: List[str]=['bond_order', 'is_in_ring', 'is_aromatic'],
        use_node_features: bool=True,
        node_features: List[str]=['element', 'charge', 'nhyd', 'hyb'],
        embedding_dim: int=128,
        embedding_use_bias: bool=True,
        projector_hidden_layers: int=2,
        projector_size: int=64,
        rescale_factor: float=1.0,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced

        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)

        # output size as
        rad_output_size = self.m_0_num_coefficients * self.sphere_channels
        self.rad_func = EdgeProjector(
            radial_basis_size=radial_basis_size,
            feature_vocab_sizes=feature_vocab_sizes,
            use_edge_features=use_edge_features,
            bond_features=bond_features,
            use_node_features=use_node_features,
            node_features=node_features,
            output_dim=rad_output_size,
            embedding_dim=embedding_dim,
            embedding_use_bias=embedding_use_bias,
            projector_hidden_layers=projector_hidden_layers,
            projector_size=projector_size
        )

        self.rescale_factor = rescale_factor

    def forward(
        self, 
        edge_distance_rbf: torch.Tensor,
        edge_index: torch.Tensor, 
        num_nodes: int, 
        feature_dict: Dict[str, torch.Tensor] = {},
        node_offset: int = 0
    ):
        """
        Forward pass for edge degree embedding.
        
        Args:
            edge_distance_rbf (torch.Tensor): Radial basis function expansion of edge distances [E, radial_basis_size]
            edge_index (torch.Tensor): Edge indices [2, E]
            num_nodes (int): Number of nodes in the graph
            feature_dict (Dict[str, torch.Tensor]): Dictionary containing node and edge features
            node_offset (int): Offset for node indices (default: 0)
            
        Returns:
            SO3_Embedding: Edge embedding in SO3 format
        """
        # Use EdgeProjector to compute edge features including distance, node features, and edge features
        x_edge_m_0 = self.rad_func(edge_distance_rbf, edge_index, feature_dict)
        
        # Reshape to [num_edges, m_0_coefficients, sphere_channels]
        x_edge_m_0 = x_edge_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        
        # Pad with zeros for higher m coefficients
        x_edge_m_pad = torch.zeros(
            (
                x_edge_m_0.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            device=x_edge_m_0.device,
        )
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        # Create SO3 embedding
        x_edge_embedding = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=x_edge_m_all.device,
            dtype=x_edge_m_all.dtype,
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(self.mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_edge_embedding._reduce_edge(edge_index[:,1] - node_offset, num_nodes)
        x_edge_embedding.embedding = x_edge_embedding.embedding / self.rescale_factor

        return x_edge_embedding