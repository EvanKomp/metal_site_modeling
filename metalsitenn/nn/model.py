# metalsitenn/nn/model.py
'''
* Author: Evan Komp
* Created: 8/20/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import Dict, List, Optional, Union
from functools import partial
from transformers.modeling_utils import PreTrainedModel
import torch
from torch import nn

from fairchem.core.models.equiformer_v2.weight_initialization import eqv2_init_weights

from .backbone import EquiformerWEdgesBackbone
from .pretrained_config import EquiformerWEdgesConfig
from .heads.node_prediction import NodePredictionHead

from ..graph_data import BatchProteinData, ModelOutput



class EquiformerWEdgesPretrainedModel(PreTrainedModel):
    """
    EquiformerWEdges backbone model wrapped into PretrainedModel.
    """
    config_class: EquiformerWEdgesConfig
    base_model_prefix = "eqwedges"

    def __init__(self, config: EquiformerWEdgesConfig):
        """
        Initialize EquiformerWEdgesPretrainedModel.
        
        Args:
            config (EquiformerWEdgesConfig): Configuration for the model.
        """
        super().__init__(config)
        self.config = config
        self._init_backbone_modules()
        self._init_heads()

    def _init_backbone_modules(self):
        raise NotImplementedError(
            "Subclasses must implement _init_sub_modules to initialize the backbone and heads."
        )
    
    def _init_heads(self):
        """
        Initialize heads for the model.
        This method should be implemented in subclasses if heads are used.
        """
        pass

    def _init_weights(self, module):
        """Initialize weights for PreTrainedModel framework compatibility."""
        eqv2_init_weights(module, weight_init=self.config.weight_init)



class EquiformerWEdgesModel(EquiformerWEdgesPretrainedModel):
    """
    EquiformerWEdges backbone model.
    """

    def _init_backbone_modules(self):
        
        self.eqwedges = EquiformerWEdgesBackbone(
            num_layers=self.config.num_layers,
            sphere_channels=self.config.sphere_channels,
            attn_hidden_channels=self.config.attn_hidden_channels,
            num_heads=self.config.num_heads,
            attn_alpha_channels=self.config.attn_alpha_channels,
            attn_value_channels=self.config.attn_value_channels,
            ffn_hidden_channels=self.config.ffn_hidden_channels,
            norm_type=self.config.norm_type,
            lmax_list=self.config.lmax_list,
            mmax_list=self.config.mmax_list,
            grid_resolution=self.config.grid_resolution,
            num_sphere_samples=None, # ignored
            edge_channels_list=self.config.edge_channels_list,
            use_m_share_rad=self.config.use_m_share_rad,
            distance_function=self.config.distance_function,
            num_distance_basis=self.config.num_distance_basis,
            attn_activation=self.config.attn_activation,
            use_s2_act_attn=self.config.use_s2_act_attn,
            use_attn_renorm=self.config.use_attn_renorm,
            ffn_activation=self.config.ffn_activation,
            use_gate_act=self.config.use_gate_act,
            use_grid_mlp=self.config.use_grid_mlp,
            use_sep_s2_act=self.config.use_sep_s2_act,
            alpha_drop=self.config.alpha_drop,
            drop_path_rate=self.config.drop_path_rate,
            proj_drop=self.config.proj_drop,
            weight_init=self.config.weight_init,
            max_radius=self.config.max_radius,
            use_time=self.config.use_time,
            film_time_embedding_dim=self.config.film_time_embedding_dim,
            film_hidden_dim=self.config.film_hidden_dim,
            film_mlp_layers=self.config.film_mlp_layers,
            film_num_gaussians=self.config.film_num_gaussians,
            film_basis_function=self.config.film_basis_function,
            feature_vocab_sizes=self.config.feature_vocab_sizes,
            atom_features=self.config.atom_features,
            bond_features=self.config.bond_features,
            edge_degree_projector_hidden_layers=self.config.edge_degree_projector_hidden_layers,
            edge_degree_projector_size=self.config.edge_degree_projector_size,
            embedding_dim=self.config.embedding_dim,
            use_topology_gradients=self.config.use_topology_gradients,
            topology_gradient_clip=self.config.topology_gradient_clip,
            avg_num_nodes=self.config.avg_num_nodes,
            avg_degree=self.config.avg_degree
        )

    def forward(
        self,
        batch: BatchProteinData,
        **kwargs
    ) -> ModelOutput:
        
        output_embeddings = self.eqwedges(
            batch,
            **kwargs
        )
        return ModelOutput(
            node_embedding=output_embeddings.embedding,
        )

class EquiformerWEdgesForPretraining(EquiformerWEdgesModel):
    """
    EquiformerWEdges model for pretraining tasks.
    This class can be extended to include specific heads for pretraining.
    """

    def _init_heads(self):
        assert 'element' in self.config.feature_vocab_sizes, "Element feature size must be defined in config."

        self.node_prediction_head = NodePredictionHead(
            backbone=self.eqwedges,
            output_dim=self.config.feature_vocab_sizes['element']
        )

        if self.config.node_class_weights is not None:
            self.register_buffer(
                'node_class_weights',
                torch.tensor(self.config.node_class_weights, dtype=torch.float32)
            )
        else:
            self.node_class_weights = None
        self.cel = nn.CrossEntropyLoss(reduction='none', weight=self.node_class_weights)

    def compute_loss(self, batch: BatchProteinData, logits: torch.Tensor, film_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the pretraining task.
        
        Args:
            batch (BatchProteinData): Batch of protein data.
            logits (torch.Tensor): Model predictions.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        cel_losses = self.cel(logits, batch.element_labels.squeeze(dim=1))

        # mask out losses for only
        if batch.atom_masked_mask is not None:
            cel_losses = cel_losses * batch.atom_masked_mask
            cel_loss = cel_losses.sum() / batch.atom_masked_mask.sum()
        else:
            cel_loss = cel_losses.mean()

        # now do the film loss if its in here
        if self.config.use_time and self.config.film_l2_loss_weight > 0.0:
            assert film_norm is not None, "Film norm must be provided when use_time is True."
            film_loss = film_norm * self.config.film_l2_loss_weight
            total_loss = cel_loss + film_loss
        else:
            film_loss = None
            total_loss = cel_loss
            

        return total_loss, cel_loss, film_loss, cel_losses
    
    def forward(
        self,
        batch: BatchProteinData,
        compute_loss: bool = False,
        return_per_node_cel_loss: bool = False,
        return_node_embedding_tensor: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass for the model.

        Args:
            batch (BatchProteinData): Batch of protein data.
            return_loss (bool): Whether to return the loss.
            return_per_node_cel_loss (bool): Whether to return per-node cross-entropy loss.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelOutput: Output containing node embeddings and film norms.
        """
        backbone_outputs = self.eqwedges(
            batch,
            **kwargs
        )
        embeddings = backbone_outputs['node_embedding']
        film_norm = backbone_outputs['film_norm']
        logits = self.node_prediction_head(embeddings)


        out_data = {'node_logits': logits}
        if return_node_embedding_tensor:
            out_data['node_embeddings'] = embeddings.embedding

        if compute_loss:
            total_loss, cel_loss, film_loss, per_node_cel_loss = self.compute_loss(
                batch, logits, film_norm
            )
            out_data['loss'] = total_loss
            out_data['node_loss'] = cel_loss
            out_data['film_l2_loss'] = film_loss
            if return_per_node_cel_loss:
                out_data['node_losses'] = per_node_cel_loss

        return ModelOutput(**out_data)



        
        
