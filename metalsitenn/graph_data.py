# metalsitenn/graph_data.py
'''
* Author: Evan Komp
* Created: 8/5/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from dataclasses import dataclass

from typing import List, Tuple, Dict, Any
import torch
import numpy as np


def make_top_k_graph(r, hop_distances, k=10):
    """
    Create a top-k graph based on bond distances.
    
    Args:
        r (torch.Tensor): Positions of atoms, shape (N, 3).
        hop_distances (torch.Tensor): Hop distances between each atom.
        k (int): Number of nearest neighbors to consider for each atom.
            Up to half are determined by bonding patterns, 
            the rest by distance.
    """
    N = r.shape[0]
    _,idx = torch.topk(hop_distances.masked_fill(hop_distances==0,999), min(k//2+1,N), largest=False)
    distance_mask = torch.zeros_like(hop_distances,dtype=bool).scatter_(1,idx,True)
    distance_mask = distance_mask & (hop_distances>0)

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

# mutable
@dataclass(frozen=False)
class ProteinData:
    """
    Data class for protein atomic data.
    N = num atoms
    E = num edges
    B = num systems in batch N

    """
    # atom info
    element: torch.Tensor=None  # [N, 1]
    charge: torch.Tensor=None  # [N, 1]
    nhyd: torch.Tensor=None  # [N, 1]
    hyb: torch.Tensor=None  # [N, 1]
    positions: torch.Tensor=None  # [N, 3]
    atom_movable_mask: torch.Tensor=None  # [N, 1] - mask for atoms that can be moved. This should contain at least all indices in atom_noised_mask

    # for posterity
    atom_name: np.ndarray[str]=None  # [N, 1] - atom names
    atom_resname: np.ndarray[str]=None  # [N, 1] - residue names
    atom_resid: torch.Tensor=None  # [N, 1] - residue ids
    atom_ishetero: torch.Tensor=None  # [N, 1] - is hetero atom

    # edge info
    distances: torch.Tensor=None  # [E, 1] - distances between atoms
    bond_order: torch.Tensor=None # [E, 1]
    is_aromatic: torch.Tensor=None  # [E, 1]
    is_in_ring: torch.Tensor=None  # [E, 1]
    edge_index: torch.Tensor=None  # [E, 2]

    # topology
    topology: Dict[str, torch.Tensor]=None  
    # Includes
    # bonds [N_bonds, 2] - pairs of atom indexes that are bonded
    # bond_lengths [N_bonds, 1] - equilibrium bond lengths of the bonds
    # planars [O,4]
    # chirals [O,4] O here is number of contraints, and 5 is the number if indexes required to specify the comstraint

    # global features
    global_features: torch.Tensor=None  # [B, d]
    time: torch.Tensor=None  # [B, 1] - time of the system, if applicable

    # attributes related to collating / loss calculation
    atom_masked_mask: torch.Tensor=None  # [N, 1] - mask for atoms that were masked
    element_labels: torch.Tensor=None  # [N, 1] - labels
    element_loss_weights: torch.Tensor=None # [N, 1] - per atom weights for element loss

    global_labels: torch.Tensor=None  # [B, d] - labels for global tasks
    global_loss_weights: torch.Tensor=None  # [B, d] - per example weights for global tasks

    atom_noised_mask: torch.Tensor=None  # [N, 1] - mask for atoms that were noised
    position_flow_labels: torch.Tensor=None  # [N, 3] - labels for positions in flow tasks
    position_labels: torch.Tensor=None  # [N, 3] - labels for positions
    position_loss_weights: torch.Tensor=None  # [N, 3] - per atom weights for denoising loss


    def __setattr__(self, name: str, value):
        """
        Override setattr to invalidate distances when positions change.
        
        Args:
            name (str): Attribute name being set
            value: Value being assigned
        """
        # Check if positions are being modified
        if name == 'positions' and hasattr(self, 'distances') and self.distances is not None:
            # Set distances to None before setting positions
            object.__setattr__(self, 'distances', None)
        
        # Call parent setattr
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        """
        Return shapes or lengths of data structures in a readable format.
        """
        repr_str = "ProteinData(\n"
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                repr_str += f"  {field}=None,\n"
            elif hasattr(value, 'shape'):
                repr_str += f"  {field}: shape={tuple(value.shape)},\n"
            elif isinstance(value, list):
                repr_str += f"  {field}: len={len(value)},\n"
            else:
                repr_str += f"  {field}={value},\n"
        repr_str += ")"
        return repr_str
    
    def set_distances(self):
        """
        Calculate and set distances based on current positions.
        
        This method computes pairwise distances between atoms and sets the
        `distances` attribute.
        """
        if self.positions is None or self.edge_index is None:
            raise ValueError("Positions must be set before calculating distances.")
        
        # Compute pairwise distances
        R = torch.cdist(self.positions, self.positions)
        # Fill diagonal with zeros (self-distances)
        R.fill_diagonal_(0.0)
        # Store distances in the ProteinData object using edge index
        src, dst = self.edge_index.t()
        distances = R[src, dst].unsqueeze(-1)  # [E, 1]
        self.distances = distances
    

class BatchProteinData:
    """
    Data class for a batch of ProteinData objects.
    Contains a list of ProteinData objects and provides methods to access
    individual ProteinData instances by index.
    """
    pass
            