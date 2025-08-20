import math
import torch
import torch.nn as nn
from typing import List, Union, Optional
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.models.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer
from fairchem.core.models.equiformer_v2.so3 import CoefficientMappingModule

from metalsitenn.nn.mlp import MLP


class SO3EquivariantFiLM(nn.Module):
    """
    SO(3)-equivariant FiLM modulation using CoefficientMappingModule for proper indexing.
    
    Args:
        coefficient_mapping: CoefficientMappingModule from equiformer for exact compatibility
        num_channels: Number of spherical channels  
        num_layers: Number of model layers (e.g., attention layers)
        time_embedding_dim: Dimension for time embedding after gaussian smearing
        hidden_dim: Hidden dimension for MLP
        mlp_layers: Number of layers in time embedding MLP
        num_gaussians: Number of gaussian basis functions for time smearing
        sigma_min: Minimum sigma for gaussian smearing (default: 1e-4)
        sigma_max: Maximum sigma for gaussian smearing (default: 1.0)
        
    Alternative constructor args (creates CoefficientMappingModule internally):
        lmax_list: List of maximum degrees for each resolution
        mmax_list: List of maximum orders for each resolution (defaults to lmax_list)
        
    Returns:
        Tensor of shape (batch_size, num_coefficients, num_channels) for single layer
        OR (num_layers, batch_size, num_coefficients, num_channels) for multi-layer
    """
    
    def __init__(
        self,
        num_channels: int,
        num_layers: int = 1,  # Default to single layer
        coefficient_mapping: Optional[CoefficientMappingModule] = None,
        lmax_list: Optional[List[int]] = None,
        mmax_list: Optional[List[int]] = None,
        time_embedding_dim: int = 256,
        hidden_dim: int = 256,
        mlp_layers: int = 3,
        basis_function: str = "gaussian",  # Type of basis function for time embedding
        num_gaussians: int = 50,
        basis_start: float = 0.0,
        basis_end: float = 1.0,
    ):
        super().__init__()
        
        # Handle coefficient mapping input
        if coefficient_mapping is not None:
            self.mapping = coefficient_mapping
            self.lmax_list = coefficient_mapping.lmax_list
            self.mmax_list = coefficient_mapping.mmax_list
        elif lmax_list is not None:
            self.lmax_list = lmax_list
            self.mmax_list = mmax_list if mmax_list is not None else lmax_list.copy()
            self.mapping = CoefficientMappingModule(self.lmax_list, self.mmax_list)
        else:
            raise ValueError("Must provide either coefficient_mapping or lmax_list")
            
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_resolutions = len(self.lmax_list)
        self.basis_function = basis_function
        
        # Gaussian time embedding
        if basis_function == "gaussian":
            self.time_embedding = GaussianSmearing(
                start=basis_start,
                stop=basis_end, 
                num_gaussians=num_gaussians,
                basis_width_scalar=(basis_end - basis_start) / num_gaussians
            )
        elif basis_function == "gaussian_rbf":
            self.time_embedding = GaussianRadialBasisLayer(
                num_basis=num_gaussians,
                cutoff=basis_end)
        else:
            raise ValueError(f"Unsupported basis function: {basis_function}. Use 'gaussian' or 'gaussian_rbf'.")
        
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

        # now we can determin the total number of gamma values
        n_gamas = self.num_layers * self.num_features * self.num_channels
        
        self.film_projection = nn.Linear(
            time_embedding_dim, 
            n_gamas, 
            bias=False
        )
    
    def _compute_coefficient_structure(self):
        """Compute coefficient structure using Equiformer's CoefficientMappingModule."""
        # Get total number of coefficients from the mapping module
        with torch.no_grad():
            # Get l and m harmonics from the mapping
            l_harmonic = self.mapping.l_harmonic
            
            self.total_coefficients = len(l_harmonic)

            # determine the number of unique features across l values in multiple resolutions
            feature_mapping = []
            f_index = 0
            last_lval = 0
            for l_val in l_harmonic:
                if l_val != last_lval:
                    f_index += 1
                    last_lval = l_val
                feature_mapping.append(f_index)

            # save feature mapping as a fixed tensor buffer
            self.num_features = f_index + 1
            self.register_buffer(
                "feature_mapping",
                torch.tensor(feature_mapping, dtype=torch.long)
            )

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
        if self.basis_function == "gaussian":
            time_embedded = self.time_embedding(time)  # (batch_size, num_gaussians)
        elif self.basis_function == "gaussian_rbf":
            time_embedded = self.time_embedding(time.squeeze(-1))  # (batch_size, num_gaussians)
        
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
        
        # Reshape gammas to (batch_size, total_features, num_channels)
        expanded_gammas = gammas.view(batch_size, self.num_features, self.num_channels)

        # index the second dimension using the feature mapping (batch_size, num_coefficients, num_channels)
        expanded_gammas = expanded_gammas.index_select(
            dim=1, 
            index=self.feature_mapping.to(device, dtype=torch.long)
        )
        return expanded_gammas

    def _expand_multi_layer(self, all_gammas: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Expand gammas for multi-layer output."""
        # here we just have the extra layer dimension that should be at the start
        device = all_gammas.device

        # Reshape to (batch_size, num_layers, total_features, num_channels)
        expanded_gammas = all_gammas.view(
            batch_size, 
            self.num_layers, 
            self.num_features, 
            self.num_channels
        )

        # expand the third dimension using the feature mapping (batch_size, num_layers, num_coefficients, num_channels)
        expanded_gammas = expanded_gammas.index_select(
            dim=2, 
            index=self.feature_mapping.to(device, dtype=torch.long)
        )

        # place the layer dimension at the start (num_layers, batch_size, num_coefficients, num_channels)
        expanded_gammas = expanded_gammas.permute(1, 0, 2, 3)
        return expanded_gammas

    def get_output_dim(self) -> int:
        """Return the total number of coefficients for validation."""
        return self.total_coefficients
        
    def extra_repr(self) -> str:
        return (f"lmax_list={self.lmax_list}, mmax_list={self.mmax_list}, "
                f"num_features={self.num_features}, "
                f"num_channels={self.num_channels}, num_layers={self.num_layers}, "
                f"total_coefficients={self.total_coefficients}")