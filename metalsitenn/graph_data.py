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
    atom_elements: torch.Tensor=None  # [N, 1]
    atom_charges: torch.Tensor=None  # [N, 1]
    atom_nhyds: torch.Tensor=None  # [N, 1]
    atom_hyb: torch.Tensor=None  # [N, 1]
    positions: torch.Tensor=None  # [N, 3]
    batch: torch.Tensor=None  # [N, 1] - used to identify which system the atom belongs to
    # for posterity
    atom_name: List[str]=None  # [N, 1] - atom names
    atom_resname: List[str]=None  # [N, 1] - residue names
    atom_resid: List[int]=None  # [N, 1] - residue ids

    # edge info
    bond_order: torch.Tensor=None # [E, 1]
    is_aromatic: torch.Tensor=None  # [E, 1]
    is_in_ring: torch.Tensor=None  # [E, 1]
    edge_index: torch.Tensor=None  # [E, 2]

    # topology
    planars: torch.Tensor=None  # [O,5]
    chirals: torch.Tensor=None  # [O,5] O here is number of contraints, and 5 is the number if indexes required to specify the comstraint

    # global features
    global_features: torch.Tensor=None  # [B, d]