

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from fairchem.core.common import gp_utils
from fairchem.core.models.equiformer_v2.transformer_block import FeedForwardNetwork
from fairchem.core.models.equiformer_v2.weight_initialization import eqv2_init_weights

class NodePredictionHead(nn.Module):
    """Head for predicting node scalars.
    
    Args:
        backbone (nn.Module): The backbone model to extract features - use parameters of the model to help determine
          internal dimensions.
        output_dim (int): The output dimension of the head.
    """
    def __init__(self, backbone: nn.Module, output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim

        # Get the internal dimension from the backbone
        internal_dim = self.backbone.sphere_channels

        # produce FFNN
        self.ffnn = FeedForwardNetwork(
            sphere_channels=internal_dim,
            hidden_channels=self.backbone.ffn_hidden_channels,
            output_channels=self.output_dim,
            lmax_list=self.backbone.lmax_list,
            mmax_list=self.backbone.mmax_list,
            SO3_grid=self.backbone.SO3_grid,
            activation=self.backbone.ffn_activation,
            use_gate_act=self.backbone.use_gate_act,
            use_grid_mlp=self.backbone.use_grid_mlp,
            use_sep_s2_act= self.backbone.use_sep_s2_act)
        
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, embeddings: 'SO3_Embeddings') -> torch.Tensor:
        """Forward pass through the node prediction head.
        
        Args:
            embeddings (SO3_Embeddings): The input embeddings from the backbone.
        
        Returns:
            torch.Tensor: The predicted node scalars.
        """
        # pass through the ffnn
        so3_tensor = self.ffnn(embeddings).embedding
        scaler_outputs = so3_tensor.narrow(dim=1, start=0, length=1) # take just the scalers
        # [N, 1, output_dim ] 
        outputs = scaler_outputs.squeeze(dim=1)  # [N, output_dim]
        return outputs
