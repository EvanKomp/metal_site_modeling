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
    batch: torch.Tensor=None  # [N, 1] - used to identify which system the atom belongs to
    # for posterity
    atom_name: List[str]=None  # [N, 1] - atom names
    atom_resname: List[str]=None  # [N, 1] - residue names
    atom_resid: List[int]=None  # [N, 1] - residue ids
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

    # attributes related to collating / loss calculation
    atom_masked_mask: torch.Tensor=None  # [N, 1] - mask for atoms that were masked
    atom_masked_labels: torch.Tensor=None  # [N, 1] - labels
    atom_loss_weights: torch.Tensor=None  # [N, 1] - loss weights for each atom
    edge_loss_weights: torch.Tensor=None  # [E, 1] - loss weights for each edge if doing edge prediction

    atom_noised_mask: torch.Tensor=None  # [N, 1] - mask for atoms that were noised
    position_labels: torch.Tensor=None  # [N, 3] - labels for positions


    def to(self, device: str):
        """
        Move all tensors in the ProteinData to the specified device.
        
        Args:
            device (str): The device to move the tensors to (e.g., 'cpu', 'cuda').
        """
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
            else:
                pass

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
            