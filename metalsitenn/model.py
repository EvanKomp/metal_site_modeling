# metalsitenn/model.py
'''
* Author: Evan Komp
* Created: 12/4/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Interface to model classes
'''

from torch import nn
import torch
from e3nn import o3
import math

from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.configuration_utils import PretrainedConfig

from metalsitenn.nn import _MetalSiteFoundationalModel, _MetalSiteNodeHead
from metalsitenn.atom_vocabulary import AtomTokenizer

from typing import List, Optional

# a helper function to generate the irreps for the model
def get_irreps(l: int, scale: int, decay: float, num_heads: int = None) -> tuple[o3.Irreps, o3.Irreps]:
    """Generate scalable irreps for network features and attention heads.
    
    Args:
        l: Maximum order of irreps
        scale: Base multiplicity for l=0 irreps
        decay: Factor to reduce multiplicity for each l (decay^l)
        num_heads: If provided, returns attention head irreps with multiplicities divided by num_heads
    
    Returns:
        hidden_irreps: Full irreps for hidden features
        head_irreps: Irreps for attention heads if num_heads provided, else None
    """
    irreps = []
    head_irreps = []
    
    for i in range(l + 1):
        # Calculate multiplicity with decay
        mult = int(scale * (decay ** i))
        if mult == 0:
            continue
            
        # Add even and odd irreps of order i
        irreps.extend([
            (mult, (i, 1)),  # even
            (mult, (i, -1))  # odd
        ])
        
        # Calculate head multiplicities if requested
        if num_heads is not None:
            head_mult = math.ceil(mult / num_heads)
            if head_mult > 0:
                head_irreps.extend([
                    (head_mult, (i, 1)),
                    (head_mult, (i, -1))
                ])
    
    hidden_irreps = o3.Irreps(irreps)
    head_irreps = o3.Irreps(head_irreps) if num_heads is not None else None
    
    return hidden_irreps, head_irreps


# EXPECTED CONFIGURATION
class MetalSiteNNConfig(PretrainedConfig):
    """Configuration for Metal site model.

    Args:
        irreps_node_embedding (str, o3.Irreps) : Irreps for node embeddings inside model layers
        irreps_sh (str, o3.Irreps) : Irreps for spherical harmonics
        irreps_foundation_output (str, o3.Irreps) : Irreps for the output of the foundation network
        irreps_pred_head_output (str, o3.Irreps) : Irreps for the output of the prediction head
        tokenizer (AtomTokenizer) : Tokenizer for atomic data, needed to get embeddings sizes
        atom_embedding_dims (int) : Dimensionality for initial atom embeddings
        max_radius (float) : Maximum radius for edges in the graph
        number_basis (int) : Number of basis functions for learned radial gaussian basis
        fc_neurons (List[int]) : Number of neurons in each fully connected layer for internal nonlinest FFNNs
        irreps_head (str, o3.Irreps) : Irreps for the output of each attention head
        num_heads (int) : Number of attention heads
        num_layers (int) : Number of transformer layers
        alpha_drop (float) : Dropout rate for attention dropout
        proj_drop (float) : Dropout rate for projection dropout
        out_drop (float) : Dropout rate for output dropout
        drop_path_rate (float) : Drop path rate edge dropping
        avg_node_degree (int) : Average number of edges per node

        output_attentions (bool, optional) : Whether to output attentions
        output_hidden_states (bool, optional) : Whether to output hidden states
        
    """

    def __init__(
        self,
        irreps_node_embedding: o3.Irreps=o3.Irreps('128x0e+64x1o+32x2e'),
        irreps_sh: o3.Irreps=o3.Irreps.spherical_harmonics(3),
        irreps_foundation_output: o3.Irreps=o3.Irreps('128x0e+64x1o+32x2e'),
        tokenizer: AtomTokenizer=None,
        atom_embedding_dims: int=16,
        max_radius: float=5.0,
        number_basis: int=32,
        fc_neurons: List[int]=[32,32],
        irreps_head: o3.Irreps=o3.Irreps('32x0e+16x1o+8x2e'),
        num_heads: int=4,
        num_layers: int=2,
        alpha_drop: float=0.0,
        proj_drop: float=0.0,
        out_drop: float=0.0,
        drop_path_rate: float=0.0,
        avg_node_degree: int=12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_starting_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_foundation_output = o3.Irreps(irreps_foundation_output)
        self.tokenizer = tokenizer
        self.atom_embedding_dims = atom_embedding_dims
        self.max_radius = max_radius
        self.number_basis = number_basis
        self.fc_neurons = fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.avg_node_degree = avg_node_degree
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_starting_embeddings = output_starting_embeddings

# expected outputs
class MetalSiteFoundationalOutput(ModelOutput):
    """
    Output of the foundational network that outputs vectors.

    Args:
        hidden_state (torch.FloatTensor) : Hidden state of the network
        node_attr (torch.FloatTensor) : Node attributes
        edge_attr (torch.FloatTensor) : Edge attributes
        edge_features (torch.FloatTensor) : Edge features
        attentions (torch.FloatTensor, optional) : Attention weights, only if output_attentions=True
        hidden_states (torch.FloatTensor, optional) : All hidden states of the network, only if output_hidden_states=True
    """

    def __init__(
        self,
        hidden_state: torch.FloatTensor,
        node_attr: torch.FloatTensor,
        edge_attr: torch.FloatTensor,
        edge_features: torch.FloatTensor,
        attentions: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
    ):
        self.hidden_state = hidden_state
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.edge_features = edge_features
        self.attentions = attentions
        self.hidden_states = hidden_states
    

class MetalSitePretrainingOutput(ModelOutput):
    """
    Output of the pretraining network that outputs vectors and logits.

    Args:
        atom_logits (torch.FloatTensor) : Logits for the classification task
        atom_type_logits (torch.FloatTensor) : Logits for the atom type classification task
        vectors (torch.FloatTensor) : Vectors for the regression task
        edge_attr (torch.FloatTensor, optional) : Edge attributes, only if output_starting_embeddings=True
        edge_features (torch.FloatTensor, optional) : Edge features, only if output_starting_embeddings=True
        node_attr (torch.FloatTensor, optional) : Node attributes, only if output_starting_embeddings=True
        attentions (torch.FloatTensor, optional) : Attention weights, only if output_attentions=True
        hidden_states (torch.FloatTensor, optional) : All hidden states of the network, only if output_hidden_states=True
        mask_ce_loss (torch.FloatTensor, optional) : Cross entropy loss for masked language model
        noise_mse_loss (torch.FloatTensor, optional) : Mean squared error loss for noise prediction
    """

    def __init__(
        self,
        atom_logits: torch.FloatTensor,
        atom_type_logits: torch.FloatTensor,
        output_vectors: torch.FloatTensor,
        edge_attr: torch.FloatTensor=None,
        edge_features: torch.FloatTensor=None,
        node_attr: torch.FloatTensor=None,
        attentions: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        mask_ce_loss: Optional[torch.FloatTensor] = None,
        noise_mse_loss: Optional[torch.FloatTensor] = None,
    ):
        self.atom_logits = atom_logits
        self.atom_type_logits = atom_type_logits
        self.output_vectors = output_vectors
        self.edge_attr = edge_attr
        self.edge_features = edge_features
        self.node_attr = node_attr
        self.attentions = attentions
        self.hidden_states = hidden_states
        self.mask_ce_loss = mask_ce_loss
        self.noise_mse_loss = noise_mse_loss
        

class MetalSitePretrainedModel(PreTrainedModel):
    """
    Base class for Metal site models
    """

    config_class = MetalSiteNNConfig
    base_model_prefix = "metalsitenn"
        
        
class MetalSiteFoundationalModel(MetalSitePretrainedModel):

    def __init__(self, config: MetalSiteNNConfig):
        super().__init__(config)
        self.config = config
        self._model = _MetalSiteFoundationalModel(
            irreps_node_embedding=config.irreps_node_embedding,
            irreps_sh=config.irreps_sh,
            irreps_output=config.irreps_node_embedding,
            tokenizer=config.tokenizer,
            atom_embedding_dims=config.atom_embedding_dims,
            max_radius=config.max_radius,
            number_basis=config.number_basis,
            fc_neurons=config.fc_neurons,
            irreps_head=config.irreps_head,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            alpha_drop=config.alpha_drop,
            proj_drop=config.proj_drop,
            out_drop=config.out_drop,
            drop_path_rate=config.drop_path_rate,
            avg_aggregate_num=config.avg_node_degree,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )

    def forward(self,
                atoms: torch.Tensor,
                atom_types: torch.Tensor,
                positions: torch.Tensor,
                batch_indices: torch.Tensor,
                **kwargs
                ) -> MetalSiteFoundationalOutput:
        # atom embedder tensor
        # in our case tokens for element and tokens for atom type
        atom_identifiers = torch.stack([atoms, atom_types], dim=-1)
        inner_output = self._model(
            atom_identifiers=atom_identifiers,
            positions=positions,
            batch_indices=batch_indices, **kwargs)
        
        # construct output object
        output = MetalSiteFoundationalOutput(
            hidden_state=inner_output['node_hidden_state'],
            node_attr=inner_output['node_attrs'],
            edge_attr=inner_output['edge_attrs'] if 'edge_attrs' in inner_output else None,
            edge_features=inner_output['edge_features'] if 'edge_features' in inner_output else None,
            attentions=inner_output['attentions'] if 'attentions' in inner_output else None,
            hidden_states=inner_output['hidden_states'] if 'hidden_states' in inner_output else None
        )
        return output


class MetalSiteForPretrainingModel(MetalSitePretrainedModel):

    def __init__(self, config: MetalSiteNNConfig):
        super().__init__(config)
        self.config = config
        self.foundational_model = MetalSiteFoundationalModel(config)
        self.pred_head = _MetalSiteNodeHead(
            irreps_node_input=config.irreps_node_embedding,
            irreps_node_attrs=self.foundational_model._model.irreps_node_attr,
            proj_drop=config.proj_drop,
            tokenizer=config.tokenizer)
        

    def compute_atom_masking_loss(
            self,
            atom_logits: torch.Tensor,
            atom_labels: torch.Tensor,
            mask_mask: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss for masked language model."""
        # get the mask
        mask = mask_mask.bool()
        # get the logits for the masked atoms
        logits = atom_logits[mask]
        # get the labels for the masked atoms
        labels = atom_labels[mask]
        # compute the loss
        return nn.functional.cross_entropy(logits, labels)
    
    def compute_position_noising_loss(
            self,
            output_vectors: torch.Tensor,
            denoise_vectors: torch.Tensor,
            noise_loss_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean squared error in euclidean distance for noising loss."""

        # get the mask
        mask = noise_loss_mask.bool()
        # get the vectors for the noised atoms
        vectors = output_vectors[mask]
        # get the denoised vectors
        denoise = denoise_vectors[mask]
        # compute the loss
        return nn.functional.mse_loss(vectors, denoise)

    def forward(
            self,
            atoms: torch.Tensor,
            atom_types: torch.Tensor,
            positions: torch.Tensor,
            batch_indices: torch.Tensor,
            mask_mask: torch.Tensor=None,
            atom_labels: torch.Tensor=None,
            atom_type_labels: torch.Tensor=None,
            noise_mask: torch.Tensor=None,
            denoise_vectors: torch.Tensor=None,
            noise_loss_mask: torch.Tensor=None,
            **kwargs) -> MetalSitePretrainingOutput:
        
        foundational_output = self.foundational_model(
            atoms=atoms,
            atom_types=atom_types,
            positions=positions,
            batch_indices=batch_indices,
            **kwargs
        )

        # apply prediction head
        atom_scalars_logits, atom_type_scalars_logits, output_vectors = self.pred_head(
            node_input=foundational_output.hidden_state,
            node_attrs=foundational_output.node_attr,
        )
        # compute losses
        if mask_mask is not None:
            mask_ce_loss = self.compute_atom_masking_loss(
                atom_logits=atom_scalars_logits,
                atom_labels=atom_labels,
                mask_mask=mask_mask
            )
        else:
            mask_ce_loss = None

        if noise_mask is not None:
            noise_mse_loss = self.compute_position_noising_loss(
                output_vectors=output_vectors,
                denoise_vectors=denoise_vectors,
                noise_loss_mask=noise_loss_mask
            )
        else:
            noise_mse_loss = None

        # construct output object
        output = MetalSitePretrainingOutput(
            atom_logits=atom_scalars_logits,
            atom_type_logits=atom_type_scalars_logits,
            output_vectors=output_vectors,
            edge_attr=foundational_output.edge_attr,
            edge_features=foundational_output.edge_features,
            node_attr=foundational_output.node_attr,
            attentions=foundational_output.attentions,
            hidden_states=foundational_output.hidden_states,
            mask_ce_loss=mask_ce_loss,
            noise_mse_loss=noise_mse_loss
        )

        return output

                
