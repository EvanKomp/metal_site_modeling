# metalsitenn/nn/topology.py
'''
* Author: Evan Komp
* Created: 8/13/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT


This module provides functions and classes for computing L1 features from molecular
topology constraints (bond lengths, chirality, planarity) and mixing them with
existing SO3 embeddings in an equivariant manner.
'''

import math
from typing import List, Optional

import torch
import torch.nn as nn

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
    """
    Get gradients of positions with respect to topology features.

    A la. ChemNet ; https://github.com/baker-laboratory/PLACER/blob/main/modules/model.py

    Some additions:
    - the gradient is flipped in direction such that it makes physical sense - these vectors point in the direction the atom should
      move. This should make no difference for downstream neural operations as weights can flip anyway.
    - option to provide mask for atoms, which will zero out gradients for masked atoms. This is useful for training with masked atoms.

    Args:
        r: Atom positions, shape (N, 3).
        bond_indexes: Bond indexes, shape (M, 2).
        bond_lengths: Bond lengths, shape (M,1).
        chirals: Chirality features, shape (O,5).
        planars: Planarity features, shape (P,5).
        gclip: Gradient clipping value.
        atom_mask: Mask for atoms, shape (N,). If provided, gradients will be zeroed for masked atoms.

    Returns:
        grads: Gradients of shape (N, 3, 3). (vectors from each of both length, chirals, planars).
    """
    N = r.shape[0]
    device = r.device

    with torch.enable_grad():
        r_detached = r.detach()  # so that the computation graph does not include the result of this function, which is essentially external context / input
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
            o, i, j, k = r_detached[chirals].permute(1, 0, 2)
            l = ((triple_prod(o-i, o-j, o-k, norm=True)-0.70710678)**2).sum()
            g[:, 1] = torch.autograd.grad(l, r_detached)[0].data

        # Compute planarity gradients
        if len(planars) > 0:
            o, i, j, k = r_detached[planars].permute(1, 0, 2)
            l = ((triple_prod(o-i, o-j, o-k, norm=True)**2).sum())
            g[:, 2] = torch.autograd.grad(l, r_detached)[0].data

        # Scale and clip
        g = torch.nan_to_num(g, nan=0.0, posinf=gclip, neginf=-gclip)
        gnorm = torch.linalg.norm(g, dim=-1)
        mask = gnorm > gclip
        g[mask] /= gnorm[mask][..., None]
        g[mask] *= gclip

        # flip direction of gradients
        g = -g

        # Zero gradients for masked atoms
        if atom_mask is not None:
            g *= atom_mask[:, None, None].to(g.dtype)

        # reorder such that the three types of gradients (channels) are the last dim, instead of the vector being there
        g = g.permute(0, 2, 1)

        return g.detach().to(r.dtype).to(r.device)


class SO3_L1_Linear(nn.Module):
    """
    Equivariant linear layer that operates only on L=1 spherical harmonic features.
    Maintains SO(3) equivariance by using shared weights for all m components.
    
    Input: [N, 3, in_channels] -> Output: [N, 3, out_channels]
    
    Args:
        in_channels: Input feature channels
        out_channels: Output feature channels  
        bias: Whether to use bias term (should be False for L=1 to maintain equivariance)
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Single weight matrix shared by all m components of L=1
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        
        if bias:
            raise ValueError("Bias should be False for L=1 to maintain equivariance")
        
    def forward(self, x):
        """
        Apply equivariant linear transformation to L=1 features.
        
        Args:
            x: [N, 3, in_channels] L=1 features (m=-1,0,1 components)
        Returns:
            [N, 3, out_channels] transformed L=1 features
        """
        # Apply same linear transformation to all m components
        # x @ weight.T maintains equivariance since all m share same weights
        return torch.einsum('nmi, oi -> nmo', x, self.weight)
    

class SO3_L1_LinearMixing(nn.Module):
    """
    Applies a linear transformation to L=1 spherical harmonic features.
    
    This layer is designed to mix L=1 features while maintaining equivariance.
    Useful for combining existing L=1 features with topology gradients.
    
    Args:
        in_channels_list: List of input sizes for each item to be mixed
        out_channels: Output size for the mixed features
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Create a linear layer for each input channel size
        self.linears = nn.ModuleList([
            SO3_L1_Linear(in_channels, out_channels) for in_channels in in_channels_list
        ])
        
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Mix multiple L=1 feature sources.
        
        Args:
            x_list: List of tensors with shape [N, 3, in_channels_i] for each input
        Returns:
            Tensor with shape [N, 3, out_channels] after mixing
        """
        mixed_features = [linear(x) for linear, x in zip(self.linears, x_list)]
        return torch.stack(mixed_features, dim=2).sum(dim=2)