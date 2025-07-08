#!/usr/bin/env python
# coding: utf-8

# # Creating neural net components based on node and 2d features
# 0. Inputs are encodings of atoms and 2d attributes (bonds)
#    - $f_a, [N, ?]$
#    - $f_b, [N,N,?]$, ? is determined by number of features and embedding size
# 1. Converting input features to scaler embeddings with eg. linear or MLP
#     - $h_a^0, [N,d^0]$ for atoms
#     - $h_b^0, [N,N,d^0]$ for bonds
# 2. Intializing SO3 embeddings with general node scaler features (unlike original implimentation which uses atomic numbers and embeds)
#     - $h_a = SO3_{init}(h_a^0), [N,so3]$
# 3. Compute graph and associated rotation matrices
#     - $G = top\_k\_graph(topology, r)$
# 4. Update invariant bond features by distance using biased attention a la chemnet
#     - $h_b = f(h_b, r)$
# 5. Incorporate initial edge information (distance RBF + bond features + initial atom embeddings)
#     - $h_a +=f(g(h_b), |r|, h_a^0), [N,so3]$
# 6. Compute node gradients a la chemnet with topology
#     - $dr = get\_gradients(topology, r), [N,3]$ This is one l=1 feature. We should output a spherical harmonic component as opposed to the vector
# 7. Into transformer blocks
#     - Layer norm 
#     - We need to write a method of incorporating additional features in the form of vectors or scalars (eg. the atom gradients)
#       - $h_a=f(h_a,dr)$, as a start, let's have learnable weight matrix of n_channels * n_resolutions (assuming each resolution is at least l=1) and apply the weights to the incoming l=1 feature, add it to each channel according to that weight
#     - We need to rewrite the attention convolution to use arbitrary edge features and distance
#       - $h_a = attention\_conv(h_a, h_b, f_b, |r|, g)$, here h_b, f_b, RBF(|R|) are concated and MLPed to be the input of the attention weights. Non weighted messages are produced by concating h_a. Weight the messages in the convolution.
# 8. Extract vectors from node embeddings and update positions, and also update pair representations
#     - $r' = get\_vectors(h_a), [N,3]$
#     - $r += r'$
# 
# -> 3-7 are repeated for a number of iterations
# 
# 9. Output final node embeddings, final bond embeddings, final positions
# 10. Prediction head
#     - FAPE like losses on final positions given frames
#     - Atom scaler embeddings -(MLP)-> vocabulary size logits for each element, CEL on atom predictions
#     - Bond embeddings (already scalar) -(MLP)-> bond order vobaulary logits, CEL on bond predictions
# 
# ***
# ***

# In[1]:


import sys

sys.path.append('/projects/metalsitenn/pdbx')

from metalsitenn.placer_modules.cifutils import CIFParser

from metalsitenn.featurizer import MetalSiteFeaturizer
from metalsitenn.utils import visualize_featurized_metal_site_3d
import pandas as pd
import numpy as np
import torch


# In[2]:


parser = CIFParser()
parsed_data = parser.parse('/datasets/alphafold_data/data_v2/pdb_mmcif/mmcif_files/6fpw.cif')
sites = parser.get_metal_sites(
    parsed_data, max_atoms_per_site=500, max_water_bfactor=15, merge_threshold=6, cutoff_distance=6, backbone_treatment='free',
    clean_metal_bonding_patterns=True)
site = sites[1]


# In[3]:


site_chain = site['site_chain']


# ## 0-1. Input encodings

# In[4]:


featurizer = MetalSiteFeaturizer(
    atom_features=['element', 'charge', 'nhyd', 'hyb'],
    bond_features=['bond_order', 'is_in_ring', 'is_aromatic']
)
features = featurizer(site_chain, metal_unknown=False)


# In[5]:


atom_features, bond_features, topology_data = features


# In[6]:


visualize_featurized_metal_site_3d(
    atom_features_dict=atom_features,
    bond_features_dict=bond_features)


# In[7]:


batch = {}
for dict in [atom_features, bond_features, topology_data]:
    for key, value in dict.items():
        if isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=torch.float32)
        batch[key] = value


# In[8]:


batch.keys()


# In[58]:


# extract some tensors we need
positions = batch['positions']
bond_distances = batch['bond_distances']
bonds = batch['bonds']
bond_lengths = batch['bond_lengths']
chirals = batch['chirals']
planars = batch['planars']


# In[10]:


import torch
import torch.nn as nn
from typing import Dict, Union, Optional, Tuple

from metalsitenn.nn.mlp import MLP


class AtomAndBondEmbedding(nn.Module):
    """
    Embedding layer for atom and bond features from MetalSiteFeaturizer.
    
    Takes tokenized atom and bond features from a single dictionary, applies learnable embeddings,
    concatenates them, and optionally applies MLPs for further processing.
    
    For bond features, also concatenates source and destination atom embeddings (after atom MLP if present).
    
    Args:
        vocab_sizes: Dictionary mapping feature names to their vocabulary sizes
                    (output from featurizer.get_feature_vocab_sizes())
        embed_dim: Embedding dimension for each feature
        mlp_hidden_size: Hidden size for both atom and bond MLPs (optional)
        mlp_n_hidden_layers: Number of hidden layers in both MLPs
        mlp_activation: Activation function for both MLPs
        mlp_dropout_rate: Dropout rate for both MLPs
        include_atom_in_bond: Whether to include atom embeddings in bond features
        
    Example:
        >>> vocab_sizes = {'element': 50, 'charge': 8, 'bond_order': 6}
        >>> embedding = AtomAndBondEmbedding(
        ...     vocab_sizes=vocab_sizes,
        ...     embed_dim=32,
        ...     mlp_hidden_size=128,
        ...     include_atom_in_bond=True
        ... )
        >>> features = {
        ...     'element': torch.randint(0, 50, (10, 1)),
        ...     'bond_order': torch.randint(0, 6, (10, 10))
        ... }
        >>> atom_concat, bond_concat, atom_hidden, bond_hidden = embedding(features)
    """
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim: int = 32,
        mlp_hidden_size: Optional[int] = None,
        mlp_n_hidden_layers: int = 2,
        mlp_activation: Union[str, nn.Module] = 'relu',
        mlp_dropout_rate: float = 0.0,
        include_atom_in_bond: bool = True
    ):
        super().__init__()
        
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        self.include_atom_in_bond = include_atom_in_bond
        
        # Separate atom and bond feature names
        # Bond features are those that would create NxN matrices
        possible_bond_features = {'bond_order', 'is_aromatic', 'is_in_ring'}
        possible_atom_features = {'element', 'charge', 'nhyd', 'hyb'}
        self.bond_feature_names = [name for name in vocab_sizes.keys() 
                                    if name in possible_bond_features]
        self.atom_feature_names = [name for name in vocab_sizes.keys()
                                   if name in possible_atom_features]
        
        # Create embeddings for each feature
        self.atom_embeddings = nn.ModuleDict()
        self.bond_embeddings = nn.ModuleDict()
        
        for feature_name, vocab_size in vocab_sizes.items():
            embedding = nn.Embedding(vocab_size, embed_dim)
            if feature_name in self.bond_feature_names:
                self.bond_embeddings[feature_name] = embedding
            else:
                self.atom_embeddings[feature_name] = embedding
        
        # Calculate concatenated embedding dimensions
        self.atom_concat_dim = len(self.atom_feature_names) * embed_dim
        self.bond_concat_dim = len(self.bond_feature_names) * embed_dim
        
        # Atom MLP (always created first if needed)
        self.atom_mlp = None
        self.atom_output_size = self.atom_concat_dim
        
        if mlp_hidden_size is not None and self.atom_concat_dim > 0:
            self.atom_mlp = MLP(
                input_size=self.atom_concat_dim,
                hidden_size=mlp_hidden_size,
                n_hidden_layers=mlp_n_hidden_layers,
                hidden_activation=mlp_activation,
                dropout_rate=mlp_dropout_rate
            )
            self.atom_output_size = mlp_hidden_size
        
        # Bond MLP (includes atom embeddings if requested)
        self.bond_mlp = None
        bond_input_size = self.bond_concat_dim
        
        if self.include_atom_in_bond and self.atom_concat_dim > 0:
            # Add 2x atom embedding size (src + dst atoms)
            bond_input_size += 2 * self.atom_output_size
        
        if mlp_hidden_size is not None and bond_input_size > 0:
            self.bond_mlp = MLP(
                input_size=bond_input_size,
                hidden_size=mlp_hidden_size,
                n_hidden_layers=mlp_n_hidden_layers,
                hidden_activation=mlp_activation,
                dropout_rate=mlp_dropout_rate
            )
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], 
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through embeddings and optional MLPs.
        
        Args:
            features: Dictionary of all features with shapes:
                     - Atom features: (N, 1) or (N,)
                     - Bond features: (N, N)
            
        Returns:
            Tuple of (atom_concat_embeddings, bond_concat_embeddings, 
                     atom_mlp_output, bond_mlp_output)
            - atom_concat_embeddings: (N, atom_concat_dim) concatenated atom embeddings
            - bond_concat_embeddings: (N, N, bond_concat_dim) concatenated bond embeddings  
            - atom_mlp_output: (N, atom_output_size) if atom_mlp exists, else None
            - bond_mlp_output: (N, N, mlp_hidden_size) if bond_mlp exists, else None
        """
        n_atoms = None
        
        # Process atom features
        atom_embeddings_list = []
        if self.atom_feature_names:
            for feature_name in self.atom_feature_names:
                if feature_name in features:
                    feature_tensor = features[feature_name]  # (N, 1) or (N,)
                    
                    # Handle different input shapes
                    if feature_tensor.dim() == 2 and feature_tensor.size(-1) == 1:
                        feature_tensor = feature_tensor.squeeze(-1)  # (N,)
                    elif feature_tensor.dim() != 1:
                        raise ValueError(f"Atom feature '{feature_name}' must have shape (N,) or (N, 1), "
                                       f"got {feature_tensor.shape}")
                    
                    if n_atoms is None:
                        n_atoms = feature_tensor.shape[0]
                    
                    # Apply embedding
                    embedded = self.atom_embeddings[feature_name](feature_tensor)  # (N, embed_dim)
                    atom_embeddings_list.append(embedded)
        
        # Process bond features
        bond_embeddings_list = []
        if self.bond_feature_names:
            for feature_name in self.bond_feature_names:
                if feature_name in features:
                    feature_tensor = features[feature_name]  # (N, N)
                    
                    if feature_tensor.dim() != 2:
                        raise ValueError(f"Bond feature '{feature_name}' must have shape (N, N), "
                                       f"got {feature_tensor.shape}")
                    
                    if n_atoms is None:
                        n_atoms = feature_tensor.shape[0]
                    
                    # Apply embedding
                    embedded = self.bond_embeddings[feature_name](feature_tensor)  # (N, N, embed_dim)
                    bond_embeddings_list.append(embedded)
        
        # Concatenate embeddings
        atom_concat = None
        bond_concat = None
        
        if atom_embeddings_list:
            atom_concat = torch.cat(atom_embeddings_list, dim=-1)  # (N, atom_concat_dim)
        
        if bond_embeddings_list:
            bond_concat = torch.cat(bond_embeddings_list, dim=-1)  # (N, N, bond_concat_dim)
        
        # Apply atom MLP first (needed for bond features)
        atom_mlp_output = None
        atom_features_for_bond = atom_concat  # Default to concatenated embeddings
        
        if self.atom_mlp is not None and atom_concat is not None:
            atom_mlp_output = self.atom_mlp(atom_concat)  # (N, mlp_hidden_size)
            atom_features_for_bond = atom_mlp_output
        
        # Apply bond MLP with optional atom embeddings
        bond_mlp_output = None
        
        if self.bond_mlp is not None and bond_concat is not None:
            bond_features_for_mlp = bond_concat
            
            # Add atom embeddings to bond features if requested
            if self.include_atom_in_bond and atom_features_for_bond is not None:
                # Create src and dst atom embedding matrices
                # src_embeddings: (N, N, atom_dim) where src_embeddings[i, j] = atom_features_for_bond[i]
                # dst_embeddings: (N, N, atom_dim) where dst_embeddings[i, j] = atom_features_for_bond[j]
                src_embeddings = atom_features_for_bond.unsqueeze(1).expand(-1, n_atoms, -1)  # (N, N, atom_dim)
                dst_embeddings = atom_features_for_bond.unsqueeze(0).expand(n_atoms, -1, -1)  # (N, N, atom_dim)
                
                # Concatenate bond features with atom embeddings
                bond_features_for_mlp = torch.cat([
                    bond_concat,          # (N, N, bond_concat_dim)
                    src_embeddings,       # (N, N, atom_output_size)
                    dst_embeddings        # (N, N, atom_output_size)
                ], dim=-1)  # (N, N, bond_concat_dim + 2*atom_output_size)
            
            # Reshape for MLP: (N, N, feature_dim) -> (N*N, feature_dim)
            original_shape = bond_features_for_mlp.shape
            bond_features_flat = bond_features_for_mlp.view(-1, original_shape[-1])
            bond_mlp_flat = self.bond_mlp(bond_features_flat)  # (N*N, mlp_hidden_size)
            bond_mlp_output = bond_mlp_flat.view(original_shape[0], original_shape[1], -1)  # (N, N, mlp_hidden_size)
        
        return atom_concat, bond_concat, atom_mlp_output, bond_mlp_output
    
    @property
    def atom_output_dim(self) -> int:
        """Get the output dimension for atom features."""
        if self.atom_mlp is not None:
            return self.atom_mlp.hidden_size
        return self.atom_concat_dim
    
    @property
    def bond_output_dim(self) -> int:
        """Get the output dimension for bond features."""
        if self.bond_mlp is not None:
            return self.bond_mlp.hidden_size
        # If no MLP, return the concatenated size including atom embeddings if used
        bond_dim = self.bond_concat_dim
        if self.include_atom_in_bond and self.atom_concat_dim > 0:
            bond_dim += 2 * self.atom_output_size
        return bond_dim
    


# In[11]:


vocab_sizes = featurizer.get_feature_vocab_sizes()
vocab_sizes


# In[12]:


atombondembedder = AtomAndBondEmbedding(
    vocab_sizes=vocab_sizes,
    embed_dim=32,
    mlp_hidden_size=128,
    mlp_n_hidden_layers=1,
    mlp_activation='relu',
    mlp_dropout_rate=0.0)


# In[13]:


atom_attributes, pair_attributes, atom_scaler_features, pair_features = atombondembedder(batch) 


# In[14]:


for tensor in [atom_attributes, pair_attributes, atom_scaler_features, pair_features]:
    print(tensor.shape)


# ## 2. Initializing SO3 node embeddings with scalar information

# In[15]:


from __future__ import annotations

import torch
import torch.nn as nn

from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding


class SO3ScalarEmbedder(nn.Module):
    """
    Converts pre-computed atom embeddings to SO3 embeddings by projecting them 
    to the l=0, m=0 coefficients across multiple resolutions.
    
    Args:
        input_dim (int): Dimension of input atom embeddings
        lmax_list (list[int]): List of maximum degrees (l) for each resolution
        sphere_channels (int): Number of spherical channels per resolution
        device (str): Device to place tensors on
    """
    
    def __init__(
        self,
        input_dim: int,
        lmax_list: list[int],
        sphere_channels: int,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.lmax_list = lmax_list
        self.sphere_channels = sphere_channels
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * sphere_channels
        
        # Projection layer to map input embeddings to spherical channels
        self.projection = nn.Linear(input_dim, self.sphere_channels_all)
        
        # Initialize weights
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, atom_embeddings: torch.Tensor) -> SO3_Embedding:
        """
        Convert atom embeddings to SO3 embeddings.
        
        Args:
            atom_embeddings (torch.Tensor): Input atom embeddings of shape (N, input_dim)
            
        Returns:
            SO3_Embedding: SO3 embedding with l=0, m=0 coefficients initialized
        """
        num_atoms = atom_embeddings.shape[0]
        
        # Project input embeddings to spherical channels
        projected_embeddings = self.projection(atom_embeddings)  # (N, sphere_channels_all)
        
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
                x.embedding[:, offset_res, :] = projected_embeddings
            else:
                # Multi-resolution case - split channels across resolutions
                x.embedding[:, offset_res, :] = projected_embeddings[
                    :, offset_channels : offset_channels + self.sphere_channels
                ]
            
            # Update offsets for next resolution
            offset_channels += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)
        
        return x


# In[17]:


so3_scaler_embedder = SO3ScalarEmbedder(
    input_dim=atombondembedder.atom_output_dim,
    lmax_list=[3],
    sphere_channels=32
)


# In[18]:


atom_features = so3_scaler_embedder(atom_scaler_features)


# In[19]:


(atom_features.embedding !=0).all(axis=-1)


# ## 3. determine graph and spacial rbf

# In[20]:


def make_top_k_graph(r, bond_distances, k=10):
    """
    Create a top-k graph based on bond distances.
    
    Args:
        r (torch.Tensor): Positions of atoms, shape (N, 3).
        bond_distances (torch.Tensor): Distances for each bond, shape (N,N).
        k (int): Number of nearest neighbors to consider for each atom.
            Up to half are determined by bonding patterns, 
            the rest by distance.
    """
    N = r.shape[0]
    _,idx = torch.topk(bond_distances.masked_fill(bond_distances==0,999), min(k//2+1,N), largest=False)
    distance_mask = torch.zeros_like(bond_distances,dtype=bool).scatter_(1,idx,True)
    distance_mask = distance_mask & (bond_distances>0)

    # then pull from actual angstrom distances
    # first compute pairwise distances|
    R = torch.cdist(r, r)  # (N, N)
    # fill in distance with the ones we have already chosen so that they are insta chosen
    R_ = R.masked_fill(distance_mask, 0.0)
    _,idx = torch.topk(R_, min(k+1,N), largest=False)
    r_mask = torch.zeros_like(R_, dtype=bool).scatter_(1, idx, True)

    # get edges
    src,dst = torch.where(r_mask.fill_diagonal_(False)) # self edge deleted
    return src, dst, R


# In[27]:


src, dst, _ = make_top_k_graph(
    positions,
    bond_distances)


# In[28]:


def get_all_atoms_for_target_atom_graph(
    src: torch.Tensor, 
    dst: torch.Tensor, 
    target_atom_idx: int
) -> torch.Tensor:
    """
    Get all atoms connected to a target atom in a graph.
    
    Args:
        src (torch.Tensor): Source indices of edges.
        dst (torch.Tensor): Destination indices of edges.
        target_atom_idx (int): Index of the target atom.
        
    Returns:
        torch.Tensor: Indices of all atoms connected to the target atom.
    """
    mask = (src == target_atom_idx)
    other_atoms = dst[mask]
    highlight_atoms = other_atoms.tolist() + [target_atom_idx]
    return highlight_atoms


# In[29]:


highlight_atoms = get_all_atoms_for_target_atom_graph(src, dst, 66)


# In[30]:


visualize_featurized_metal_site_3d(
    atom_features_dict={'element': batch['element'],
                        'positions': batch['positions'],},
    bond_features_dict={'bond_order': batch['bond_order'],},
    highlight_atoms=highlight_atoms)


# In[31]:


from fairchem.core.models.equiformer_v2.so3 import SO3_Rotation, CoefficientMappingModule, SO3_Grid
from fairchem.core.models.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.models.equiformer_v2.module_list import ModuleListInfo

class EquivarianceSupport(nn.Module):
    def __init__(self, 
                 lmax_list: list[int], 
                 mmax_list: list[int],
                 top_k: int = 10,
                 rbf_start: float = 0.0,
                 rbf_stop: float = 20.0,
                 rbf_num_gaussians: int = 64,
                 rbf_basis_width_scalar: float = 1.0,
                 grid_resolution: int | None = None,
                 grid_normalization: str = "component"):
        """Organizes equivariance support for SO3 embeddings.
        
        Args:
            lmax_list (list[int]): List of maximum degrees (l) for each resolution.
            mmax_list (list[int]): List of maximum orders (m) for each resolution.
            top_k (int): Number of neighbors, prioritizing bonds and then distances.
            rbf_start (float): Start value for Gaussian RBF centers.
            rbf_stop (float): Stop value for Gaussian RBF centers.
            rbf_num_gaussians (int): Number of Gaussian basis functions.
            rbf_basis_width_scalar (float): Width scaling for Gaussian basis functions.
            grid_resolution (int | None): Resolution for SO3_Grid. If None, uses default based on lmax.
            grid_normalization (str): Normalization type for SO3_Grid ("component" or "integral").
        """
        super().__init__()
        
        # Core equivariance components (nn.Module subcomponents)
        # IMPORTANT: SO3_Rotation uses only lmax, not mmax!
        self._SO3_rotation = nn.ModuleList([SO3_Rotation(lmax) for lmax in lmax_list])
        self._mappingReduced = CoefficientMappingModule(lmax_list, mmax_list)
        
        # Initialize SO3_Grid following the original EquiformerV2 pattern
        max_lmax = max(lmax_list)
        self._SO3_grid = ModuleListInfo(f"({max_lmax}, {max_lmax})")
        
        for lval in range(max_lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max_lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=grid_resolution,
                        normalization=grid_normalization,
                    )
                )
            self._SO3_grid.append(SO3_m_grid)
        
        # Gaussian RBF module
        self.rbf = GaussianSmearing(
            start=rbf_start,
            stop=rbf_stop,
            num_gaussians=rbf_num_gaussians,
            basis_width_scalar=rbf_basis_width_scalar
        )
        
        # Configuration
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.top_k = top_k
        self.grid_resolution = grid_resolution
        self.grid_normalization = grid_normalization
        
        # Graph structure storage
        self.register_buffer('_edge_index', torch.empty(0), persistent=False)
        
        # Distance matrices (full N x N matrices)
        self.register_buffer('_distance_matrix', torch.empty(0), persistent=False)  # 3D spatial distances
        self.register_buffer('_distance_matrix_rbf', torch.empty(0), persistent=False)  # RBF of spatial distances
        
        # Edge-specific data (derived from matrices using edge_index)
        self.register_buffer('_edge_distance_vec', torch.empty(0), persistent=False) 
        self.register_buffer('_edge_distance', torch.empty(0), persistent=False)  # spatial distance for edges
        self.register_buffer('_edge_distance_rbf', torch.empty(0), persistent=False)  # RBF for edges
        self.register_buffer('_edge_rot_mat', torch.empty(0), persistent=False)
        
        # State tracking
        self._graph_valid = False
        self._num_atoms = 0

    def _check_valid(self):
        """Check if graph data is valid, raise error if not."""
        if not self._graph_valid:
            raise RuntimeError(
                "Graph data is not valid. Call update_graph() first."
            )

    def update_graph(self, positions: torch.Tensor, hop_distances: torch.Tensor, **kwargs):
        """
        Update all graph-related data and rotation matrices.
        
        Args:
            positions: [N, 3] atom positions
            hop_distances: [N, N] bond hop distances (number of bonds between atoms)
            **kwargs: additional arguments for your graph function
        """
        self._num_atoms = positions.shape[0]
        
        # Generate new src, dst edges and get distance matrix from graph function
        src, dst, distance_matrix = make_top_k_graph(positions, hop_distances, self.top_k)
        
        # Apply RBF to the full distance matrix
        # Flatten, apply RBF, then reshape back
        distance_flat = distance_matrix.view(-1)  # [N*N]
        distance_rbf_flat = self.rbf(distance_flat)  # [N*N, num_gaussians]
        distance_matrix_rbf = distance_rbf_flat.view(
            self._num_atoms, self._num_atoms, -1
        )  # [N, N, num_gaussians]
        
        # Create edge_index tensor [2, num_edges]
        edge_index = torch.stack([src, dst], dim=0)
        
        # Compute edge vectors and distances by indexing into positions
        edge_distance_vec = positions[dst] - positions[src]  # [num_edges, 3]
        
        # Get edge distances by indexing into the distance matrix
        edge_distance = distance_matrix[src, dst]  # [num_edges]
        
        # Get edge RBF features by indexing into the RBF distance matrix
        edge_distance_rbf = distance_matrix_rbf[src, dst]  # [num_edges, num_gaussians]
        
        # Compute rotation matrices for each edge
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)  # [num_edges, 3, 3]
        
        # Update stored tensors
        self._edge_index = edge_index
        self._distance_matrix = distance_matrix
        self._distance_matrix_rbf = distance_matrix_rbf
        self._edge_distance_vec = edge_distance_vec
        self._edge_distance = edge_distance
        self._edge_distance_rbf = edge_distance_rbf
        self._edge_rot_mat = edge_rot_mat
        
        # Update Wigner-D matrices in SO3_rotation modules
        for i in range(len(self.lmax_list)):
            self._SO3_rotation[i].set_wigner(edge_rot_mat)
        
        # Mark as valid
        self._graph_valid = True

    def get_edge_data(self) -> Dict[str, torch.Tensor]:
        """
        Returns all edge-related data as a dictionary.
        
        Returns:
            dict with keys: 'edge_index', 'edge_distance', 'edge_distance_vec', 
                           'edge_distance_rbf', 'edge_rot_mat'
        """
        self._check_valid()
        return {
            'edge_index': self._edge_index,
            'edge_distance': self._edge_distance,
            'edge_distance_vec': self._edge_distance_vec,
            'edge_distance_rbf': self._edge_distance_rbf,
            'edge_rot_mat': self._edge_rot_mat
        }

    def get_distance_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Returns the full distance matrices.
        
        Returns:
            dict with keys: 'distance_matrix', 'distance_matrix_rbf'
        """
        self._check_valid()
        return {
            'distance_matrix': self._distance_matrix,
            'distance_matrix_rbf': self._distance_matrix_rbf
        }

    def get_pairwise_distance(self, i: int, j: int) -> torch.Tensor:
        """
        Get spatial distance between atoms i and j.
        
        Args:
            i, j: atom indices
            
        Returns:
            scalar tensor with distance between atoms i and j
        """
        self._check_valid()
        return self._distance_matrix[i, j]

    def get_pairwise_distance_rbf(self, i: int, j: int) -> torch.Tensor:
        """
        Get RBF features for spatial distance between atoms i and j.
        
        Args:
            i, j: atom indices
            
        Returns:
            tensor of shape [num_gaussians] with RBF features
        """
        self._check_valid()
        return self._distance_matrix_rbf[i, j]

    def rotate_embedding_forward(self, embedding: 'SO3_Embedding') -> 'SO3_Embedding':
        """
        Rotate SO3_Embedding to edge frame (forward rotation).
        This follows the original Equiformer pattern.
        """
        self._check_valid()
        rotated_embedding = embedding.clone()
        rotated_embedding._rotate(
            self._SO3_rotation, 
            self.lmax_list, 
            self.mmax_list
        )
        return rotated_embedding
        
    def rotate_embedding_inverse(self, embedding: 'SO3_Embedding') -> 'SO3_Embedding':
        """
        Rotate SO3_Embedding back to global frame (inverse rotation).
        This follows the original Equiformer pattern.
        """
        self._check_valid()
        rotated_embedding = embedding.clone()
        rotated_embedding._rotate_inv(
            self._SO3_rotation, 
            self._mappingReduced
        )
        return rotated_embedding

    def is_valid(self) -> bool:
        """Check if graph data is valid/up-to-date."""
        return self._graph_valid

    def invalidate(self):
        """Mark graph data as invalid (useful for debugging/explicit control)."""
        self._graph_valid = False

    @property
    def edge_index(self) -> torch.Tensor:
        self._check_valid()
        return self._edge_index

    @property  
    def edge_distance_vec(self) -> torch.Tensor:
        self._check_valid()
        return self._edge_distance_vec

    @property
    def edge_distance(self) -> torch.Tensor:
        """Get spatial distances for edges only."""
        self._check_valid() 
        return self._edge_distance

    @property
    def distance_matrix(self) -> torch.Tensor:
        """Get full spatial distance matrix [N, N]."""
        self._check_valid()
        return self._distance_matrix
    
    @property
    def edge_distance_rbf(self) -> torch.Tensor:
        """Get RBF-transformed edge distances."""
        self._check_valid()
        return self._edge_distance_rbf

    @property
    def distance_matrix_rbf(self) -> torch.Tensor:
        """Get full RBF-transformed distance matrix [N, N, num_gaussians]."""
        self._check_valid()
        return self._distance_matrix_rbf

    @property
    def edge_rot_mat(self) -> torch.Tensor:
        self._check_valid()
        return self._edge_rot_mat

    @property
    def num_atoms(self) -> int:
        """Get number of atoms in the current graph."""
        self._check_valid()
        return self._num_atoms

    # Direct access to core components
    @property
    def SO3_rotation(self) -> nn.ModuleList:
        return self._SO3_rotation

    @property  
    def mappingReduced(self) -> 'CoefficientMappingModule':
        return self._mappingReduced
    
    @property
    def num_rbf_features(self) -> int:
        """Get the number of RBF features (useful for downstream layers)."""
        return self.rbf.num_output
    
    @property
    def SO3_grid(self) -> ModuleListInfo:
        return self._SO3_grid


# In[32]:


equivariance_support = EquivarianceSupport(
    lmax_list=[3],
    mmax_list=[2],
    top_k=10
)
equivariance_support.update_graph(
    positions=positions,
    hop_distances=bond_distances
)


# In[33]:


equivariance_support._SO3_rotation


# In[34]:


equivariance_support.edge_distance_rbf.shape


# In[35]:


equivariance_support.distance_matrix_rbf.shape


# ## 4. Update invariant bond features by distance

# In[36]:


from metalsitenn.placer_modules.pair_update import PairStr2Pair


# In[37]:


pair_update1 = PairStr2Pair(
    d_pair=atombondembedder.bond_output_dim,
    n_head=4, 
    d_hidden=32,
    d_rbf=equivariance_support.num_rbf_features)


# In[39]:


pair_features_ = pair_update1(
    pair=pair_features,
    rbf_feat=equivariance_support.distance_matrix_rbf)


# In[40]:


(pair_features_ != pair_features).all()


# In[41]:


pair_features = pair_features_


# In[42]:


# store residuals for later
pair_features_residual = pair_features.clone()


# ## 5. Incorporate initial edge information
# 
# The original EdgeDegreeEmbedding stores SO3 Rotation, Mapping Reduce - we will update it to store the initiated EquivarianceSupport instead. Additionally for edge features it uses distance RBF plus optionally atomic indices which in creates new embeddings for. We want to be able to pass an arbitrary set of edge features that we compute from distance RBF, and BOND embeddings, which already have initial atom embeddings in them as well.

# In[43]:


from fairchem.core.models.equiformer_v2.radial_function import RadialFunction


class EdgeDegreeEmbedding(torch.nn.Module):
    """
    Edge degree embedding that uses EquivarianceSupport for rotation handling.
    
    Args:
        sphere_channels (int): Number of spherical channels
        equivariance_support: Pre-initialized EquivarianceSupport instance
        edge_features_dim (int): Dimension of input edge features
        edge_channels_list (list[int]): List of sizes for edge embedding network
        rescale_factor (float): Rescale factor for sum aggregation
        use_distances (bool): Whether to include distance RBF features. If False,
                             assumes edge_features are already fully processed.
    """

    def __init__(
        self,
        sphere_channels: int,
        equivariance_support,
        edge_features_dim: int,
        edge_channels_list: list[int],
        use_distances: bool = False,
    ):
        super().__init__()
        
        self.sphere_channels = sphere_channels
        self.equivariance_support = equivariance_support
        self.lmax_list = equivariance_support.lmax_list
        self.mmax_list = equivariance_support.mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.rescale_factor = self.equivariance_support.top_k
        self.use_distances = use_distances

        # Get coefficient mapping
        self.mappingReduced = equivariance_support.mappingReduced
        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficients = len(self.mappingReduced.l_harmonic)
        
        print(f"Debug: m_0 coefficients: {self.m_0_num_coefficients}")
        print(f"Debug: Total mapping coefficients: {self.m_all_num_coefficients}")

        # Edge feature processing network
        output_dim = self.m_0_num_coefficients * self.sphere_channels
        
        # Determine input dimension based on use_distances flag
        if self.use_distances:
            input_dim = edge_features_dim + equivariance_support.num_rbf_features
        else:
            input_dim = edge_features_dim
        
        self.edge_channels_list = edge_channels_list.copy()
        self.edge_channels_list.insert(0, input_dim)
        self.edge_channels_list.append(output_dim)
        
        self.edge_feature_network = RadialFunction(self.edge_channels_list)

    def forward(
        self,
        pair_features: torch.Tensor,
        node_offset: int = 0,
    ):
        """Forward pass following the original Equiformer pattern.
        
        pair_features: Tensor of pairwise features of shape [N, N, d]
        """
        num_nodes = pair_features.shape[0]
        # Get edge data
        edge_data = self.equivariance_support.get_edge_data()
        src_idx = edge_data['edge_index'][0]
        dst_idx = edge_data['edge_index'][1]
        
        # Extract edge features
        if pair_features.dim() == 2:
            edge_feat = pair_features[src_idx, dst_idx].unsqueeze(-1)
        elif pair_features.dim() == 3:
            edge_feat = pair_features[src_idx, dst_idx]
        else:
            raise ValueError(f"edge_features must be 2D or 3D, got shape {pair_features.shape}")
        
        # Optionally combine with distance features
        if self.use_distances:
            edge_distances = edge_data['edge_distance_rbf']
            combined_features = torch.cat([edge_feat, edge_distances], dim=-1)
        else:
            # Use edge features as-is, assuming they're already fully processed
            combined_features = edge_feat
        
        # Process through network
        x_edge_m_0 = self.edge_feature_network(combined_features)
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.sphere_channels)
        
        # Pad with zeros for higher-order coefficients
        num_higher_coeffs = self.m_all_num_coefficients - self.m_0_num_coefficients
        if num_higher_coeffs > 0:
            x_edge_m_pad = torch.zeros(
                x_edge_m_0.shape[0],
                num_higher_coeffs,
                self.sphere_channels,
                device=x_edge_m_0.device,
                dtype=x_edge_m_0.dtype
            )
            x_edge_m_all = torch.cat([x_edge_m_0, x_edge_m_pad], dim=1)
        else:
            x_edge_m_all = x_edge_m_0
        
        # Create SO3_Embedding using the exact pattern from original EdgeDegreeEmbedding
        x_edge_embedding = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=x_edge_m_all.device,
            dtype=x_edge_m_all.dtype,
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape to l-primary layout
        x_edge_embedding._l_primary(self.mappingReduced)

        # Rotate back to global frame
        x_edge_embedding._rotate_inv(
            self.equivariance_support.SO3_rotation, 
            self.mappingReduced
        )

        # Reduce edges to nodes
        x_edge_embedding._reduce_edge(dst_idx - node_offset, num_nodes)
        x_edge_embedding.embedding = x_edge_embedding.embedding / self.rescale_factor

        return x_edge_embedding


# In[44]:


edge_deg_embedder = EdgeDegreeEmbedding(
    sphere_channels=32,
    equivariance_support=equivariance_support,
    edge_features_dim=atombondembedder.bond_output_dim,
    edge_channels_list=[64],
)


# In[45]:


edge_deg_embedder


# In[46]:


edge_deg_embeddings = edge_deg_embedder(
    pair_features=pair_features,
    node_offset=0,
)


# In[47]:


edge_deg_embeddings.embedding.shape


# In[48]:


edge_deg_embeddings.embedding == 0


# In[49]:


atom_features.embedding.shape


# In[50]:


atom_features.embedding = atom_features.embedding + edge_deg_embeddings.embedding


# In[52]:


# norm them now even though they wull be normed again in the Attention layer
# this is because we will be ading gradient L1 features between now and then
# and we want those learned vectors to be added to a normed space
# also its from this point we will be storing residuals
from fairchem.core.models.equiformer_v2.layer_norm import get_normalization_layer
norm_atom_inputs = get_normalization_layer(
    'layer_norm_sh',
    lmax = max(equivariance_support.lmax_list),
    num_channels=atom_features.num_channels,
)


# In[53]:


atom_features.embedding = norm_atom_inputs(atom_features.embedding)


# In[54]:


atom_features_residual = atom_features.embedding.clone()


# We now have scaler embeddings of each atom as well as l>0 features initialized with pairwise information

# ## 6. Compute node gradients
# Here we are compute gradients per atom in space (eg. not loss gradients) to supply as L1 features to the atom embeddings. Inspired by chemnet - then do a weighted sum of those L1 features

# In[55]:


np.array(batch['atom_resname'])[batch['planars']]


# In[56]:


from metalsitenn.placer_modules.losses import bondLoss
from metalsitenn.placer_modules.geometry import triple_prod

def compute_positional_topology_gradients(
    r: torch.Tensor,
    bond_indexes: torch.Tensor,
    bond_lengths: torch.Tensor,
    chirals: torch.Tensor,
    planars: torch.Tensor,
    gclip: float = 100.0,
    atom_mask: Optional[torch.Tensor] = None,
):
    """Get gradients of positions with respect to topology features.

    A la. ChemNet ; https://github.com/baker-laboratory/PLACER/blob/main/modules/model.py

    Some additions:
    - the gradient is flipped in direction such that it makes physical sense - these vectors point in the direction the atom should
      move. This should make no difference for downstream neural operations as weights can flip anyway.
    - option to provide mask for atoms, which will zero out gradients for masked atoms. This is useful for training with masked atoms.

    Args:
        r (torch.Tensor): Atom positions, shape (N, 3).
        bonds (torch.Tensor): Bond indexes, shape (M, 2).
        bond_lengths (torch.Tensor): Bond lengths, shape (M,1).
        chirals (torch.Tensor): Chirality features, shape (O,5).
        planars (torch.Tensor): Planarity features, shape (P,5).
        gclip (float): Gradient clipping value.
        atom_mask (torch.Tensor, optional): Mask for atoms, shape (N,). If provided, gradients will be zeroed for masked atoms.

    Returns:
        grads (torch.Tensor): Gradients of shape (N, 3, 3). (vectors from each of both length, chirals, planars).
    """
    N = r.shape[0]
    device = r.device

    with torch.enable_grad():
        r_detached = r.detach() # so that the computation graph does not include the result of this function, which is essentially external context / input
        r_detached.requires_grad = True  # Enable gradients for positions

        g = torch.zeros((N, 3, 3), device=device)
    
        # Compute bond gradients
        if len(bond_indexes) > 0:
            l = bondLoss(
                r_detached,
                ij=bond_indexes,
                b0=bond_lengths,
                mean=False
            )
            g[:, 0] = torch.autograd.grad(l, r_detached)[0].data

        # Compute chirality gradients
        if len(chirals) > 0:
            o,i,j,k = r_detached[chirals].permute(1, 0, 2)
            l = ((triple_prod(o-i,o-j,o-k,norm=True)-0.70710678)**2).sum()
            g[:, 1] = torch.autograd.grad(l, r_detached)[0].data

        # Compute planarity gradients
        if len(planars) > 0:
            o,i,j,k = r_detached[planars].permute(1, 0, 2)
            l = ((triple_prod(o-i,o-j,o-k,norm=True)**2).sum())
            g[:, 2] = torch.autograd.grad(l, r_detached)[0].data

        # Scale and clip
        g = torch.nan_to_num(g, nan=0.0, posinf=gclip, neginf=-gclip)
        gnorm = torch.linalg.norm(g, dim=-1)
        mask = gnorm > gclip
        g[mask] /= gnorm[mask][...,None]
        g[mask]  *= gclip

        # flip direction of gradients
        g = -g

        # Zero gradients for masked atoms
        if atom_mask is not None:
            g *= atom_mask[:, None, None].to(g.dtype)

        return g.detach()
    


# In[60]:


position_grads =compute_positional_topology_gradients(
    r=positions,
    bond_indexes=bonds,
    bond_lengths=bond_lengths,
    chirals=chirals,
    planars=planars,
)


# In[61]:


visualize_featurized_metal_site_3d(
    atom_features_dict={
        'element': batch['element'],
        'positions': batch['positions'],
    },
    bond_features_dict={
        'bond_order': batch['bond_order'],
    },
    velocities=position_grads[:,0,:])


# In[63]:


from e3nn import o3

class InjectVectors(nn.Module):
    def __init__(
        self,
        num_vectors: int,
        channels: int,
        lmax_list: list[int],
    ):
        super().__init__()
        
        self.num_vectors = num_vectors
        self.channels = channels
        self.lmax_list = lmax_list
        
        # Create linear layers only for L=1 coefficients in resolutions that support L>=1
        self.l1_linear_layers = nn.ModuleList()
        
        for lmax in self.lmax_list:
            if lmax >= 1:
                # Input: original L=1 channels + injected vector channels
                # Output: same number of channels
                linear_layer = nn.Linear(
                    self.channels + self.num_vectors, 
                    self.channels, 
                    bias=False
                )
                self.l1_linear_layers.append(linear_layer)
            else:
                self.l1_linear_layers.append(None)
        
        # Precompute L=1 coefficient indices for each resolution
        self.l1_indices = []
        offset = 0
        for lmax in self.lmax_list:
            if lmax >= 1:
                l1_start = offset + 1  # After L=0
                l1_end = offset + 4    # L=1 has 3 components
                self.l1_indices.append((l1_start, l1_end))
            else:
                self.l1_indices.append(None)
            
            offset += (lmax + 1) ** 2
    
    def forward(self, node_embedding: SO3_Embedding, vectors: torch.Tensor) -> SO3_Embedding:
        """
        Args:
            node_embedding: SO3_Embedding [N, coeffs, channels]
            vectors: Tensor [N, num_vectors, 3] - raw xyz vectors
        """
        output_embedding = node_embedding.clone()
        new_embedding_data = node_embedding.embedding.clone()
        
        # Process L=1 coefficients for each resolution
        for res_idx, (l1_indices, linear_layer) in enumerate(zip(self.l1_indices, self.l1_linear_layers)):
            if l1_indices is not None and linear_layer is not None:
                l1_start, l1_end = l1_indices
                
                # Extract original L=1 features [N, 3, channels]
                l1_input = node_embedding.embedding[:, l1_start:l1_end, :]
                
                # Reorder vectors to match spherical harmonic ordering (y, z, x)
                # vectors: [N, num_vectors, 3] (x, y, z)
                vectors_reordered = vectors[:, :, [1, 2, 0]]  # (y, z, x)
                vectors_reordered = vectors_reordered.transpose(1, 2)  # [N, 3, num_vectors]
                
                # Concatenate along channel dimension
                l1_combined = torch.cat([l1_input, vectors_reordered], dim=-1)
                
                # Apply linear transformation
                l1_output = linear_layer(l1_combined)
                
                # Update L=1 coefficients
                new_embedding_data[:, l1_start:l1_end, :] = l1_output
        
        output_embedding.set_embedding(new_embedding_data)
        return output_embedding


# In[65]:


injector = InjectVectors(
    num_vectors=position_grads.shape[1],
    channels=atom_features.num_channels,
    lmax_list=atom_features.lmax_list
)


# In[66]:


atom_features_injected = injector(
    atom_features, position_grads)


# In[67]:


(atom_features_injected.embedding[3] != atom_features.embedding[3]).any()


# ## 7. Into transformer blocks

# #### before we iteratively update node embeddings, we need to prepare edge features from pair features which will remain constant for all blocks

# In[68]:


atom_attributes.shape


# In[69]:


USE_ATOM_ATTRIBUTES = True

src, dst = equivariance_support.edge_index
edge_features = torch.cat([pair_features[src, dst], equivariance_support.edge_distance_rbf], dim=-1)

if USE_ATOM_ATTRIBUTES:
    # Concatenate atom features to edge features
    atom_attr_src = atom_attributes[src]
    atom_attr_dst = atom_attributes[dst]
    edge_features = torch.cat([edge_features, atom_attr_src, atom_attr_dst], dim=-1)
    edge_feature_dims = atombondembedder.bond_output_dim + equivariance_support.num_rbf_features + 2 * atombondembedder.atom_concat_dim
else:
    edge_feature_dims = atombondembedder.bond_output_dim + equivariance_support.num_rbf_features


# #### Class for convolutions, assuming an arbitrary set of edge l=0 features

# In[70]:


from fairchem.core.common import gp_utils

from fairchem.core.models.equiformer_v2.activation import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
import copy
import math

from fairchem.core.models.equiformer_v2.radial_function import RadialFunction
from fairchem.core.models.equiformer_v2.so2_ops import SO2_Convolution
from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from fairchem.core.models.equiformer_v2.drop import EquivariantDropoutArraySphericalHarmonics, GraphDropPath
from fairchem.core.models.equiformer_v2.transformer_block import FeedForwardNetwork

import torch_geometric

class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_channels (int):  Number of channels for alpha vector in each attention head
        attn_value_channels (int):  Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        
        equivariance_support:       EquivarianceSupport instance containing SO3_rotation, mappingReduced, etc.

        edge_feature_dim (int):     Dimensionality of input edge scalar features
        edge_channels_list (list:int):  List of sizes for edge feature processing. For example, [edge_feature_dim, hidden_channels, hidden_channels].
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        output_channels: int,
        equivariance_support,  # EquivarianceSupport instance
        edge_feature_dim: int,
        edge_channels_list: list[int],
        use_m_share_rad: bool = False,
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
        
        # Get attributes from equivariance_support
        self.equivariance_support = equivariance_support
        self.lmax_list = equivariance_support.lmax_list
        self.mmax_list = equivariance_support.mmax_list
        self.num_resolutions = len(self.lmax_list)

        # Access the core components through equivariance_support
        self.SO3_rotation = equivariance_support.SO3_rotation
        self.mappingReduced = equivariance_support.mappingReduced
        self.SO3_grid = equivariance_support.SO3_grid

        # Edge feature processing
        self.edge_feature_dim = edge_feature_dim
        self.use_m_share_rad = use_m_share_rad
        
        # Build the complete edge channels list for RadialFunction
        # Input: edge_feature_dim -> Hidden layers -> Output size (depends on use_m_share_rad)
        hidden_channels_list = copy.deepcopy(edge_channels_list) if edge_channels_list else []
        
        # Determine the output size based on whether we're using shared radial functions
        if self.use_m_share_rad:
            # For shared radial functions, output size is based on spherical harmonics
            output_size = 2 * self.sphere_channels * (max(self.lmax_list) + 1)
        else:
            # For SO2_Convolution, it will use this list internally
            # The output size will be determined by the SO2_Convolution itself
            output_size = hidden_channels_list[-1] if hidden_channels_list else self.edge_feature_dim
        
        # Construct the full edge channels list: [input, *hidden, output]
        self.edge_channels_list = [self.edge_feature_dim] + hidden_channels_list + [output_size]

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
            # For shared radial functions, create the radial function with the full edge channels list
            self.rad_func = RadialFunction(self.edge_channels_list)
            
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)
            
            # For SO2_Convolution when using shared radial functions, pass None
            so2_edge_channels_list = None
        else:
            # For non-shared radial functions, SO2_Convolution will handle edge processing internally
            so2_edge_channels_list = self.edge_channels_list

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=so2_edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

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
        edge_features: torch.Tensor,
        node_offset: int = 0,
    ):
        """
        Args:
            x: SO3_Embedding - Node embeddings
            edge_features: torch.Tensor - Arbitrary edge scalar features [num_edges, edge_feature_dim]
            node_offset: int - Node offset for graph parallel processing
        """
        # Check that the equivariance support has valid graph data
        if not self.equivariance_support.is_valid():
            raise RuntimeError(
                "EquivarianceSupport does not have valid graph data. "
                "Call update_graph() on equivariance_support before using attention."
            )
            
        # Get edge index from equivariance support
        edge_index = self.equivariance_support.edge_index
        
        # Use the provided edge features directly
        x_edge = edge_features

        x_source = x.clone()
        x_target = x.clone()
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region(x.embedding, dim=0)
            x_source.set_embedding(x_full)
            x_target.set_embedding(x_full)
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

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
    


class TransBlock(torch.nn.Module):
    """
    Updated TransformerBlock that uses arbitrary scalar edge features and EquivarianceSupport.

    Args:
        sphere_channels (int):      Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_channels (int):  Number of channels for alpha vector in each attention head
        attn_value_channels (int):  Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        equivariance_support:       EquivarianceSupport instance containing SO3_rotation, mappingReduced, SO3_grid, etc.

        edge_feature_dim (int):     Dimensionality of input edge scalar features
        edge_channels_list (list:int):  List of sizes for edge feature processing. For example, [edge_feature_dim, hidden_channels, hidden_channels].
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh'])

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN
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
        equivariance_support,  # EquivarianceSupport instance
        edge_feature_dim: int,
        edge_channels_list: list[int],
        use_m_share_rad: bool = False,
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

        # Get attributes from equivariance_support
        self.equivariance_support = equivariance_support
        lmax_list = equivariance_support.lmax_list
        mmax_list = equivariance_support.mmax_list
        SO3_grid = equivariance_support.SO3_grid
        
        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            equivariance_support=equivariance_support,
            edge_feature_dim=edge_feature_dim,
            edge_channels_list=edge_channels_list,
            use_m_share_rad=use_m_share_rad,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = (
            EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False)
            if proj_drop > 0.0
            else None
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

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

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(
                sphere_channels, output_channels, lmax=max_lmax
            )
        else:
            self.ffn_shortcut = None

    def forward(
        self,
        x,  # SO3_Embedding
        edge_features: torch.Tensor,  # [num_edges, edge_feature_dim] - arbitrary scalar edge features
        batch=None,  # for GraphDropPath
        node_offset: int = 0,
    ):
        """
        Forward pass using arbitrary edge features instead of atomic numbers and distances.
        
        Args:
            x: SO3_Embedding - Node embeddings
            edge_features: torch.Tensor - Arbitrary scalar edge features [num_edges, edge_feature_dim]
            batch: Batch tensor for GraphDropPath
            node_offset: Node offset for graph parallel processing
        """
        # if batch is non, zeros for each atom
        if batch is None:
            batch = torch.zeros(x.embedding.shape[0], device=x.embedding.device, dtype=torch.long)

        output_embedding = x.clone()

        # First residual connection: attention
        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        
        # Use the new attention that takes arbitrary edge features
        output_embedding = self.ga(
            output_embedding, 
            edge_features,  # Pass arbitrary edge features instead of atomic_numbers, edge_distance, edge_index
            node_offset
        )

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        output_embedding.embedding = output_embedding.embedding + x_res

        # Second residual connection: FFN
        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device,
                dtype=output_embedding.dtype,
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(
                output_embedding.lmax_list.copy(),
                output_embedding.lmax_list.copy(),
            )
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding


# In[71]:


trans_block = TransBlock(
    sphere_channels=atom_features.num_channels,
    attn_hidden_channels=64,
    num_heads=4,
    attn_alpha_channels=16,
    attn_value_channels=16,
    ffn_hidden_channels=64,
    output_channels=atom_features.num_channels,
    equivariance_support=equivariance_support,
    edge_feature_dim=edge_feature_dims,
    edge_channels_list=[64])


# In[72]:


trans_block


# In[73]:


atom_features = trans_block(
    atom_features_injected,
    edge_features)


# ## 8. Extract vectors from node embeddings and update positions, and also update pair representations

# In[74]:


vector_head = TransBlock(
    sphere_channels=atom_features.num_channels,
    attn_hidden_channels=64,
    num_heads=4,
    attn_alpha_channels=16,
    attn_value_channels=16,
    ffn_hidden_channels=64,
    output_channels=1,  # Force head outputs 3D forces
    equivariance_support=equivariance_support,
    edge_feature_dim=edge_feature_dims,
    edge_channels_list=[64],
)


# In[75]:


vector_outs = vector_head(
    atom_features,
    edge_features)


# In[76]:


vector_outs = vector_outs.embedding.narrow(1,1,3).view(-1,3)
# convert to cartesian
vector_outs = vector_outs[:, [2, 0, 1]]


# In[77]:


DX_SCALE = 1
# update positions
# but skip gradients back to the original gradients
new_positions = positions.detach() + DX_SCALE * vector_outs


# In[78]:


visualize_featurized_metal_site_3d(
    atom_features_dict={
        'element': batch['element'],
        'positions': positions,
    },
    bond_features_dict={
        'bond_order': batch['bond_order'],
    },
    velocities=vector_outs.detach())


# In[79]:


visualize_featurized_metal_site_3d(
    atom_features_dict={
        'element': batch['element'],
        'positions': new_positions.detach()
    },
    bond_features_dict={
        'bond_order': batch['bond_order'],
    })


# ### now update pair represenation with the new positions

# In[81]:


pair_update2 = PairStr2Pair(
    d_pair=atombondembedder.bond_output_dim,
    n_head=4, 
    d_hidden=32,
    d_rbf=equivariance_support.num_rbf_features)


# In[82]:


pair_features.shape


# In[84]:


pair_features = pair_features_residual + pair_update2(
    pair=pair_features,
    rbf_feat=equivariance_support.distance_matrix_rbf)


# ### Also update atom features with residuals from all the way at the beginning

# In[86]:


atom_features.embedding = atom_features.embedding + atom_features_residual


# ### Not update the graph with the new positions

# In[87]:


equivariance_support.update_graph(
    positions=new_positions,
    hop_distances=bond_distances)


# ### Put the recycler together

# In[ ]:


class EquivariantUpdate(nn.Module):
    """
    Wrapper module that encapsulates one iteration of the equivariant update process:
    1. Compute positional topology gradients
    2. Inject L1 vector features into atom embeddings
    3. Update atom features through transformer block
    4. Extract position updates from atom features
    5. Update positions and pair features
    6. Update graph with new positions
    
    This module is designed to be called iteratively with the same weights (recycled).
    """
    
    def __init__(
        self,
        # Core components
        equivariance_support,
        atom_embed_dim: int,
        pair_embed_dim: int,
        edge_feature_dim: int,
        atom_attributes_dim: int,
        
        # Gradient computation params
        gradient_clip: float = 100.0,
        position_update_scale: float = 1.0,
        
        # Vector injection params
        num_gradient_types: int = 3,  # bond, chiral, planar gradients
        
        # Transformer block params
        num_layers: int = 2,
        sphere_channels: int = 32,
        attn_hidden_channels: int = 64,
        num_heads: int = 4,
        attn_alpha_channels: int = 16,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 64,
        edge_channels_list: Optional[list] = [64],
        use_m_share_rad: bool = False,
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        norm_type: str = "layer_norm_sh",
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        
        # Pair update params
        pair_n_head: int = 4,
        pair_d_hidden: int = 32,
        
        # Feature combination params
        use_atom_attributes: bool = True,
    ):
        super().__init__()
        
        self.equivariance_support = equivariance_support
        self.gradient_clip = gradient_clip
        self.position_update_scale = position_update_scale
        self.use_atom_attributes = use_atom_attributes
        
        # Default edge channels list if not provided
        if edge_channels_list is None:
            edge_channels_list = [64]

        # norm of pair and atom features
        self.atom_norm = get_normalization_layer(
            norm_type, lmax=max(equivariance_support.lmax_list), num_channels=atom_embed_dim
        )
        self.pair_norm = nn.LayerNorm(pair_embed_dim)
        
        # Vector injection module
        self.vector_injector = InjectVectors(
            num_vectors=num_gradient_types,
            channels=sphere_channels,
            lmax_list=equivariance_support.lmax_list,
        )
        
        # Main transformer block for atom feature updates
        self.atom_transformer = []

        for _ in range(num_layers):
            self.atom_transformer.append(TransBlock(
                sphere_channels=sphere_channels,
                attn_hidden_channels=attn_hidden_channels,
                num_heads=num_heads,
                attn_alpha_channels=attn_alpha_channels,
                attn_value_channels=attn_value_channels,
                ffn_hidden_channels=ffn_hidden_channels,
                output_channels=sphere_channels,
                equivariance_support=equivariance_support,
                edge_feature_dim=edge_feature_dim,
                edge_channels_list=edge_channels_list.copy(),
                use_m_share_rad=use_m_share_rad,
                use_s2_act_attn=use_s2_act_attn,
                use_attn_renorm=use_attn_renorm,
                ffn_activation=ffn_activation,
                use_gate_act=use_gate_act,
                use_grid_mlp=use_grid_mlp,
                use_sep_s2_act=use_sep_s2_act,
                norm_type=norm_type,
                alpha_drop=alpha_drop,
                drop_path_rate=drop_path_rate,
                proj_drop=proj_drop,
            ))
        
        # Vector head for position updates
        self.vector_head = TransBlock(
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            output_channels=1,  # Output 3D vectors
            equivariance_support=equivariance_support,
            edge_feature_dim=edge_feature_dim,
            edge_channels_list=edge_channels_list.copy(),
            use_m_share_rad=use_m_share_rad,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            ffn_activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            norm_type=norm_type,
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
        )
        
        # Pair update module
        self.pair_updater = PairStr2Pair(
            d_pair=pair_embed_dim,
            n_head=pair_n_head,
            d_hidden=pair_d_hidden,
            d_rbf=equivariance_support.num_rbf_features,
        )
        
        # Store dimensions for edge feature construction
        self.pair_embed_dim = pair_embed_dim
        self.atom_attributes_dim = atom_attributes_dim if use_atom_attributes else 0
        

    def construct_edge_features(
        self,
        pair_features: torch.Tensor,
        atom_attributes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct edge features from pair features and optional atom attributes."""
        src, dst = self.equivariance_support.edge_index
        
        # Start with pair features and distance RBF
        edge_features = torch.cat([
            pair_features[src, dst], 
            self.equivariance_support.edge_distance_rbf
        ], dim=-1)
        
        # Optionally add atom attributes
        if self.use_atom_attributes and atom_attributes is not None:
            atom_attr_src = atom_attributes[src]
            atom_attr_dst = atom_attributes[dst]
            edge_features = torch.cat([edge_features, atom_attr_src, atom_attr_dst], dim=-1)
        
        return edge_features

    def forward(
        self,
        atom_features: SO3_Embedding,
        pair_features: torch.Tensor,
        positions: torch.Tensor,
        bond_distances: torch.Tensor,
        bond_indexes: torch.Tensor,
        bond_lengths: torch.Tensor,
        chirals: torch.Tensor,
        planars: torch.Tensor,
        atom_attributes: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[SO3_Embedding, torch.Tensor, torch.Tensor]:
        """
        Perform one iteration of the equivariant update.
        
        Args:
            atom_features: Current SO3 atom embeddings
            pair_features: Current pair features [N, N, d_pair]
            positions: Current atom positions [N, 3]
            bond_distances: Bond hop distances [N, N]
            bond_indexes: Bond indices [M, 2]
            bond_lengths: Target bond lengths [M, 1]
            chirals: Chiral constraints [O, 5]
            planars: Planar constraints [P, 5]
            atom_attributes: Optional atom attributes [N, d_atom] for edge features
            atom_mask: Optional mask for gradient computation [N]
            batch: Optional batch tensor for dropout
            
        Returns:
            Tuple of (updated_atom_features, updated_pair_features, updated_positions)
        """
        # Store residuals
        atom_features_residual = atom_features.embedding.clone()
        pair_features_residual = pair_features.clone()

        # norm inputs
        atom_features.embedding = self.atom_norm(atom_features.embedding)
        pair_features = self.pair_norm(pair_features)
        
        # 1. Compute positional topology gradients
        position_grads = compute_positional_topology_gradients(
            positions=positions,
            bond_indexes=bond_indexes,
            bond_lengths=bond_lengths,
            chirals=chirals,
            planars=planars,
            atom_mask=atom_mask,
            gradient_clip=self.gradient_clip,
            atom_mask=atom_mask,
        )
        
        # 2. Inject L1 vector features into normed atom embeddings
        # they will get normed again in the transformer blocks
        atom_features_injected = self.vector_injector(atom_features, position_grads)
        
        # 3. Construct edge features
        edge_features = self.construct_edge_features(pair_features, atom_attributes)
        
        # 4. Update atom features through transformer block
        updated_atom_features = atom_features_injected.clone()
        for layer in self.atom_transformer:
            updated_atom_features = self.atom_transformer(
                updated_atom_features,
                edge_features,
                batch=batch,
            )
        
        
        # 5. Extract position updates
        vector_outs = self.vector_head(updated_atom_features, edge_features, batch=batch)
        vector_outs = vector_outs.embedding.narrow(1, 1, 3).view(-1, 3)
        # Convert from spherical harmonic ordering (y, z, x) to cartesian (x, y, z)
        vector_outs = vector_outs[:, [2, 0, 1]]
        
        # 6. Update positions
        new_positions = positions.detach() + self.position_update_scale * vector_outs
        
        # 7. Update graph with new positions
        self.equivariance_support.update_graph(
            positions=new_positions,
            hop_distances=bond_distances,
        )
        
        # 8. Update pair features
        updated_pair_features = pair_features_residual + self.pair_updater(
            pair=pair_features,
            rbf_feat=self.equivariance_support.distance_matrix_rbf,
        )
        
        # 9. Add residual connections
        updated_atom_features.embedding = updated_atom_features.embedding + atom_features_residual
        
        return updated_atom_features, updated_pair_features, new_positions

