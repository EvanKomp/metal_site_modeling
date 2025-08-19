import math
import torch
import torch.nn as nn
from typing import List, Union, Optional
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.models.equiformer_v2.so3 import CoefficientMappingModule

from metalsitenn.nn.mlp import MLP


class SO3EquivariantFiLM(nn.Module):
    """
    SO(3)-equivariant FiLM modulation using CoefficientMappingModule for proper indexing.
    
    Args:
        lmax_list: List of maximum degrees for each resolution
        num_channels: Number of spherical channels
        num_layers: Number of model layers (e.g., attention layers)
        mmax_list: List of maximum orders for each resolution (defaults to lmax_list)
        time_embedding_dim: Dimension for time embedding after gaussian smearing
        hidden_dim: Hidden dimension for MLP
        mlp_layers: Number of layers in time embedding MLP
        num_gaussians: Number of gaussian basis functions for time smearing
        sigma_min: Minimum sigma for gaussian smearing (default: 1e-4)
        sigma_max: Maximum sigma for gaussian smearing (default: 1.0)
        
    Returns:
        Tensor of shape (batch_size, num_coefficients, num_channels) for single layer
        OR (num_layers, batch_size, num_coefficients, num_channels) for multi-layer
    """
    
    def __init__(
        self,
        lmax_list: List[int],
        num_channels: int,
        num_layers: int = 1,  # Default to single layer
        mmax_list: List[int] = None,
        time_embedding_dim: int = 128,
        hidden_dim: int = 256,
        mlp_layers: int = 3,
        num_gaussians: int = 50,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
    ):
        super().__init__()
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list if mmax_list is not None else lmax_list.copy()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_resolutions = len(lmax_list)
        
        # Use Equiformer's coefficient mapping for exact compatibility
        self.mapping = CoefficientMappingModule(self.lmax_list, self.mmax_list)
        
        # Gaussian time embedding
        self.time_embedding = GaussianSmearing(
            start=sigma_min,
            stop=sigma_max, 
            num_gaussians=num_gaussians
        )
        
        # Time encoding MLP
        self.time_mlp = MLP(
            input_size=num_gaussians,
            hidden_size=hidden_dim,
            n_hidden_layers=mlp_layers - 1,
            hidden_activation='silu',
            output_activation=None,
            dropout_rate=0.0
        )
        
        # Final projection to time_embedding_dim if different from hidden_dim
        if hidden_dim != time_embedding_dim:
            self.time_proj = nn.Linear(hidden_dim, time_embedding_dim, bias=False)
        else:
            self.time_proj = nn.Identity()
        
        # Compute coefficient structure using Equiformer's mapping
        self._compute_coefficient_structure()
        
        # Projection for gamma parameters
        # For each (resolution, l), we need num_channels gamma values
        self.total_l_channels = sum(
            num_channels * (self.lmax_list[res_idx] + 1) 
            for res_idx in range(self.num_resolutions)
        )
        
        if self.num_layers == 1:
            # Single layer output
            self.film_projection = nn.Linear(
                time_embedding_dim, 
                self.total_l_channels, 
                bias=False
            )
        else:
            # Multi-layer output
            self.film_projection = nn.Linear(
                time_embedding_dim, 
                self.num_layers * self.total_l_channels, 
                bias=False
            )
    
    def _compute_coefficient_structure(self):
        """Compute coefficient structure using Equiformer's CoefficientMappingModule."""
        # Get total number of coefficients from the mapping module
        with torch.no_grad():
            # Create dummy tensors to determine structure
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
            self.mapping.device = device
            
            # Get l and m harmonics from the mapping
            l_harmonic = self.mapping.l_harmonic.to(device)
            m_harmonic = self.mapping.m_harmonic.to(device)
            
            self.total_coefficients = len(l_harmonic)
            
            # Create mapping from coefficient index to (resolution, l, m)
            self.coeff_to_res_l_m = []
            self.res_l_to_coeff_indices = {}
            self.res_l_to_proj_slice = {}
            
            # Track where each resolution starts
            offset = 0
            proj_offset = 0
            
            for res_idx in range(self.num_resolutions):
                lmax = self.lmax_list[res_idx]
                mmax = self.mmax_list[res_idx]
                
                res_coeffs = 0
                for l in range(lmax + 1):
                    # Count coefficients for this l in this resolution
                    actual_mmax = min(mmax, l)
                    num_m = 2 * actual_mmax + 1
                    
                    # Find coefficient indices for this (res, l)
                    coeff_indices = []
                    for m_idx in range(-actual_mmax, actual_mmax + 1):
                        coeff_indices.append(offset + res_coeffs)
                        self.coeff_to_res_l_m.append((res_idx, l, m_idx))
                        res_coeffs += 1
                    
                    self.res_l_to_coeff_indices[(res_idx, l)] = coeff_indices
                    
                    # Projection slice for this (res, l)
                    self.res_l_to_proj_slice[(res_idx, l)] = slice(
                        proj_offset, proj_offset + self.num_channels
                    )
                    proj_offset += self.num_channels
                
                offset += res_coeffs
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SO(3)-equivariant FiLM.
        
        Args:
            time: Time values of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            gamma: Modulation tensor 
                   Shape (batch_size, num_coefficients, num_channels) if num_layers=1
                   Shape (num_layers, batch_size, num_coefficients, num_channels) if num_layers>1
        """
        batch_size = time.shape[0]
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        
        # a) Gaussian smearing of time
        time_embedded = self.time_embedding(time)  # (batch_size, num_gaussians)
        
        # b) Pass through MLP  
        time_features = self.time_mlp(time_embedded)  # (batch_size, hidden_dim)
        time_features = self.time_proj(time_features)  # (batch_size, time_embedding_dim)
        
        # c) Project to gamma parameters
        all_gammas = self.film_projection(time_features)  # (batch_size, total_l_channels * num_layers)
        
        if self.num_layers == 1:
            return self._expand_single_layer(all_gammas, batch_size)
        else:
            return self._expand_multi_layer(all_gammas, batch_size)
    
    def _expand_single_layer(self, gammas: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Expand gammas for single layer output."""
        device = gammas.device
        dtype = gammas.dtype
        
        # Initialize output tensor
        gamma_output = torch.zeros(
            (batch_size, self.total_coefficients, self.num_channels),
            dtype=dtype, device=device
        )
        
        # Fill in each (res, l) combination
        for res_idx in range(self.num_resolutions):
            lmax = self.lmax_list[res_idx]
            for l in range(lmax + 1):
                # Extract gamma for this (res, l)
                proj_slice = self.res_l_to_proj_slice[(res_idx, l)]
                l_gamma = gammas[:, proj_slice]  # (batch_size, num_channels)
                
                # Get coefficient indices for this (res, l)
                coeff_indices = self.res_l_to_coeff_indices[(res_idx, l)]
                
                # Broadcast: same gamma for all m values within this (res, l)
                for coeff_idx in coeff_indices:
                    gamma_output[:, coeff_idx, :] = l_gamma
        
        return gamma_output
    
    def _expand_multi_layer(self, all_gammas: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Expand gammas for multi-layer output."""
        # Reshape to separate layers
        all_gammas = all_gammas.view(
            batch_size, self.num_layers, self.total_l_channels
        )  # (batch_size, num_layers, total_l_channels)
        
        layer_gammas = []
        for layer_idx in range(self.num_layers):
            layer_gamma = all_gammas[:, layer_idx, :]  # (batch_size, total_l_channels)
            expanded_gamma = self._expand_single_layer(layer_gamma, batch_size)
            layer_gammas.append(expanded_gamma)
        
        # Stack to get shape (num_layers, batch_size, num_coefficients, num_channels)
        return torch.stack(layer_gammas, dim=0)
    
    def get_output_dim(self) -> int:
        """Return the total number of coefficients for validation."""
        return self.total_coefficients
        
    def extra_repr(self) -> str:
        return (f"lmax_list={self.lmax_list}, mmax_list={self.mmax_list}, "
                f"num_channels={self.num_channels}, num_layers={self.num_layers}, "
                f"total_coefficients={self.total_coefficients}")