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
from dataclasses import dataclass
import numpy as np

from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.configuration_utils import PretrainedConfig
from torch_scatter import scatter

from metalsitenn.nn import MetalSiteFoundationalBackbone, MetalSiteNodeHeadLayer

from typing import List, Optional

import logging
logger = logging.getLogger(__name__)

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
    """Configuration for E3-equivariant metal site model.
    
    Args:
        irreps_node_feats: o3.Irreps = '128x0e+64x1o+32x2e' 
            Initial node feature irreps
        irreps_sh: o3.Irreps = o3.Irreps.spherical_harmonics(3)
            Spherical harmonic irreps for edges
        irreps_node_output: o3.Irreps = '128x0e+64x1o+32x2e'
            Final node feature irreps
        atom_vocab_size: int=14
            Number of atom types, including mask etc.
        atom_type_vocab_size: int=3
            Number of atom types, including mask, etc.
        atom_embed_dim: int = 16
            Dimension of atomic embeddings
        max_radius: float = 5.0 
            Maximum radius (Å) for edges
        num_basis: int = 32
            Number of radial basis functions
        fc_neurons: List[int] = [32, 32]
            Hidden layer sizes for attention networks
        irreps_head: o3.Irreps = '32x0e+16x1o+8x2e'
            Feature irreps per attention head
        num_heads: int = 4
            Number of attention heads
        num_layers: int = 2 
            Number of transformer blocks
        alpha_drop: float = 0.0
            Attention dropout rate
        proj_drop: float = 0.0
            Projection dropout rate
        drop_path_rate: float = 0.0
            Skip connection dropout rate
        avg_num_neighbors: int = 12
            Expected neighbors for degree embedding
        output_attentions: bool = False
            Return attention weights
        output_hidden_states: bool = False
            Return all hidden states
    """
    def __init__(
        self,
        irreps_node_feats: o3.Irreps = o3.Irreps('128x0e+64x1o+32x2e'),
        irreps_sh: o3.Irreps = o3.Irreps.spherical_harmonics(3),
        irreps_node_output: o3.Irreps = o3.Irreps('128x0e+64x1o+32x2e'),
        atom_vocab_size: int = 14,
        atom_type_vocab_size: int = 3,
        atom_embed_dim: int = 16,
        max_radius: float = 5.0,
        num_basis: int = 32,
        fc_neurons: List[int] = [32,32],
        irreps_head: o3.Irreps = o3.Irreps('32x0e+16x1o+8x2e'),
        num_heads: int = 4,
        num_layers: int = 2,
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        avg_num_neighbors: int = 12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_initial_embeddings: bool = False,
        label_smoothing_factor: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
            
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_sh = o3.Irreps(irreps_sh)  
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.atom_vocab_size = atom_vocab_size
        self.atom_type_vocab_size = atom_type_vocab_size
        self.atom_embed_dim = atom_embed_dim
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.fc_neurons = fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.drop_path_rate = drop_path_rate
        self.avg_num_neighbors = avg_num_neighbors
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_initial_embeddings = output_initial_embeddings
        self.label_smoothing_factor = label_smoothing_factor

    def to_dict(self):
        """Convert config to dict, handling non-serializable objects"""
        config_dict = super().to_dict()
        
        # Convert irreps to strings
        config_dict["irreps_node_feats"] = str(self.irreps_node_feats)
        config_dict["irreps_sh"] = str(self.irreps_sh)
        config_dict["irreps_node_output"] = str(self.irreps_node_output)
        config_dict["irreps_head"] = str(self.irreps_head)
        
        # Remove tokenizer from serialization
        config_dict.pop("tokenizer", None)
        
        return config_dict
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dict, handling irreps strings"""
        # Convert irreps strings back to Irreps objects
        for key in ["irreps_node_feats", "irreps_sh", "irreps_node_output", "irreps_head"]:
            if key in config_dict:
                config_dict[key] = o3.Irreps(config_dict[key])
                
        return super().from_dict(config_dict)

# expected outputs
@dataclass
class MetalSiteFoundationalOutput(ModelOutput):
    """Output of the foundational backbone network.
    
    Args:
        node_feats: [num_nodes, irreps_node_output.dim] Node features
        node_attr: [num_nodes, irreps_node_attr.dim] Node attributes  
        edge_attr: [num_edges, irreps_edge_attr.dim] Edge spherical harmonics
        edge_embedding: [num_edges, num_basis] Edge radial features
        attentions: Optional List[[num_nodes, num_heads, num_nodes]] 
            Attention weights per layer
        hidden_states: Optional List[[num_nodes, irreps_node_feats.dim]]
            Node features from each layer
    """
    node_feats: torch.FloatTensor = None
    node_attr: torch.FloatTensor = None
    edge_attr: torch.FloatTensor = None
    edge_embedding: torch.FloatTensor = None
    attentions: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    
@dataclass
class MetalSitePretrainingOutput(ModelOutput):
    """Output of the pretraining model.
    
    Args:
        atom_logits: [num_nodes, atom_vocab_size] Atom type predictions
        type_logits: [num_nodes, record_vocab_size] Record type predictions
        output_vectors: [num_nodes, 3] Predicted coordinate vectors
        edge_attr: Optional [num_edges, irreps_edge_attr.dim] Edge spherical harmonics
        edge_embedding: Optional [num_edges, num_basis] Edge radial features 
        node_attr: Optional [num_nodes, irreps_node_attr.dim] Node attributes
        attentions: Optional List[[num_nodes, num_heads, num_nodes]] Attention weights
        hidden_states: Optional List[[num_nodes, irreps_node_feats.dim]] Node features
        mask_loss: Optional [] Masked atom prediction loss (sum)
        noise_loss: Optional [] Coordinate prediction loss (sum)
    """
    atom_logits: torch.FloatTensor = None
    type_logits: torch.FloatTensor = None
    output_vectors: torch.FloatTensor = None
    edge_attr: Optional[torch.FloatTensor] = None
    edge_embedding: Optional[torch.FloatTensor] = None
    node_attr: Optional[torch.FloatTensor] = None
    attentions: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    mask_loss: Optional[torch.FloatTensor] = None
    noise_loss: Optional[torch.FloatTensor] = None
     

class MetalSitePretrainedModel(PreTrainedModel):
    """
    Base class for Metal site models
    """

    config_class = MetalSiteNNConfig
    base_model_prefix = "metalsitenn"
        
        
class MetalSiteFoundationalModel(MetalSitePretrainedModel):
    """Wrapper for E3-equivariant transformer backbone."""
    
    def __init__(self, config: MetalSiteNNConfig):
        super().__init__(config)
        self.config = config
        self.backbone = MetalSiteFoundationalBackbone(
            irreps_node_feats=config.irreps_node_feats,
            irreps_sh=config.irreps_sh,
            irreps_node_output=config.irreps_node_output,
            atom_vocab_size=config.atom_vocab_size,
            atom_type_vocab_size=config.atom_type_vocab_size,
            atom_embed_dim=config.atom_embed_dim,
            max_radius=config.max_radius,
            num_basis=config.num_basis,
            fc_neurons=config.fc_neurons,
            irreps_head=config.irreps_head,
            num_heads=config.num_heads, 
            num_layers=config.num_layers,
            alpha_drop=config.alpha_drop,
            proj_drop=config.proj_drop,
            drop_path_rate=config.drop_path_rate,
            avg_num_neighbors=config.avg_num_neighbors,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )

    def forward(
        self,
        atoms: torch.Tensor,
        atom_types: torch.Tensor, 
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        **kwargs
    ) -> MetalSiteFoundationalOutput:
        """Forward pass through foundational model.
        
        Args:
            atoms: [num_nodes] Atom type tokens
            atom_types: [num_nodes] Record type tokens
            pos: [num_nodes, 3] Node coordinates
            batch_idx: [num_nodes] Batch assignments
            
        Returns:
            MetalSiteFoundationalOutput containing node features and attributes
        """
        # Stack tokens for embedding
        atom_tokens = torch.stack([atoms, atom_types], dim=-1)

        # Forward through backbone 
        outputs = self.backbone(
            atom_tokens=atom_tokens,
            pos=pos, 
            batch_idx=batch_idx,
            **kwargs
        )
        
        return MetalSiteFoundationalOutput(**outputs)
    
    def embed_systems(
            self,
            atoms,
            atom_types,
            pos,
            batch_idx,
            tokenizer,
            atom_labels: Optional[torch.Tensor] = None,
            hidden_state: int = -1,
            how: str='<METAL>_mean', **kwargs):
        """Embed atomic data for a set of systems.
        
        Args:
            atoms: [num_atoms] Atom type tokens
            atom_types: [num_atoms] Record type tokens
            pos: [num_atoms, 3] Node coordinates
            batch_idx: [num_atoms] Batch assignments
            tokenizer: AtomTokenizer
                Tokenizer for embeddings - needed to de-tokenize and see which tokens to use
            how: str = '<METAL>_mean'
                Aggregation method for system embeddings
                First string is 'all' or a particular token to consider for embeddings
                Second string an aggregation method eg. 'mean', 'sum', 'max'
            atom_labels: Optional [num_atoms] True atom tokens. Used for indexing if available
            hidden_state: int = -1
                Hidden state index to use for embeddings
        Returns:
            MetalSiteFoundationalOutput containing node features and attributes
        """
        # set such that hidden states are returned
        self.backbone.output_hidden_states = True

        # Forward through backbone
        outputs = self.forward(
            atoms=atoms,
            atom_types=atom_types,
            pos=pos,
            batch_idx=batch_idx,
            **kwargs
        )

        # Extract hidden states
        hidden_states = outputs.hidden_states[hidden_state]
        # get only scalar embeddings
        # the others are higher order tensors
        embedding_indexes = self.backbone.irreps_node_output.ls == 0

        # Get embeddings for atoms and atom types
        hidden_states = hidden_states[:, embedding_indexes]
        if atom_labels is not None:
            tokens_to_index = atom_labels
        else:
            tokens_to_index = atoms

        # de tokenize the embeddings
        token_strings = tokenizer.decode(atoms=tokens_to_index)
        # get the embeddings for the tokens
        token_to_target, mean_type = how.split('_')
        if token_to_target == 'all':
            pass
        else:
            # ensure token is in tokenizer
            assert token_to_target in tokenizer.atom_vocab, f"{token_to_target} not in tokenizer"
            # get embeddings for only the target token
            correct_token_mask = token_strings == token_to_target
            hidden_states = hidden_states[correct_token_mask]
            batch_idx = batch_idx[correct_token_mask]

        # Aggregate embeddings
        if mean_type == 'mean':
            embeddings = scatter(hidden_states, batch_idx, reduce='mean')
        elif mean_type == 'sum':
            embeddings = scatter(hidden_states, batch_idx, reduce='sum')
        elif mean_type == 'max':
            embeddings = scatter(hidden_states, batch_idx, reduce='max')
        else:
            raise ValueError(f"Invalid aggregation method: {mean_type}")
        
        self.backbone.output_hidden_states = False
        
        return embeddings



class MetalSiteForPretrainingModel(MetalSitePretrainedModel):
    """E3-equivariant transformer model for metal site pretraining."""

    def __init__(self, config: MetalSiteNNConfig):
        super().__init__(config)
        self.config = config
        self.backbone = MetalSiteFoundationalModel(config)
        self.head = MetalSiteNodeHeadLayer(
            irreps_node_feats=config.irreps_node_feats,
            irreps_node_attr=self.backbone.backbone.irreps_node_attr,
            proj_drop=config.proj_drop,
            atom_vocab_size=config.atom_vocab_size,
            atom_type_vocab_size=config.atom_type_vocab_size,
        )
        self.cel = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing_factor,
                                       reduction='none') 

    def compute_mask_loss(self, atom_logits, atom_labels, mask_indices, batch_idx):
        """Loss normalized per system"""
        mask = mask_indices.bool()
        # Compute loss per atom
        per_atom_loss = self.cel(
            atom_logits[mask],
            atom_labels[mask],
        )
        # Mean within each system then mean across systems
        # removes bias of large systems
        system_losses = scatter(per_atom_loss, batch_idx[mask], reduce='mean')
        return system_losses.mean()
    
    def compute_noise_loss(self, output_vectors, target_vectors, loss_indices, batch_idx):
        mask = loss_indices.bool()
        per_atom_loss = torch.sqrt(((output_vectors[mask] - target_vectors[mask]) ** 2).sum(dim=-1))
        system_losses = scatter(per_atom_loss, batch_idx[mask], reduce='mean') 
        return system_losses.mean()
    
    def forward(
            self,
            atoms: torch.Tensor,
            atom_types: torch.Tensor,
            pos: torch.Tensor,
            batch_idx: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            atom_labels: Optional[torch.Tensor] = None,
            atom_type_labels: Optional[torch.Tensor] = None,
            noise_mask: Optional[torch.Tensor] = None,
            denoise_vectors: Optional[torch.Tensor] = None,
            **kwargs
        ) -> MetalSitePretrainingOutput:
        """Forward pass through pretraining model.
        
        Args:
            atoms: [num_nodes] Atom type tokens
            atom_types: [num_nodes] Record type tokens 
            pos: [num_nodes, 3] Node coordinates
            batch_idx: [num_nodes] Batch assignments
            mask: Optional [num_nodes] Boolean mask of masked atoms
            atom_labels: Optional [num_nodes] True atom tokens at masked positions
            atom_type_labels: Optional [num_nodes] True type tokens at masked positions
            noise_mask: Optional [num_nodes] Boolean mask of noised coordinates
            denoise_vectors: Optional [num_nodes, 3] Target coordinate vectors
        """
        backbone_out = self.backbone(
            atoms=atoms,
            atom_types=atom_types,
            pos=pos,
            batch_idx=batch_idx,
            **kwargs
        )

        atom_logits, type_logits, output_vectors = self.head(
            node_feats=backbone_out.node_feats,
            node_attr=backbone_out.node_attr,
        )

        # Compute losses if labels provided
        mask_loss = None
        if mask is not None and atom_labels is not None:
            mask_loss = self.compute_mask_loss(
                atom_logits=atom_logits,
                atom_labels=atom_labels,
                mask_indices=mask,  # Pass mask indices
                batch_idx=batch_idx
            )
            mask_loss += self.compute_mask_loss(
                atom_logits=type_logits,
                atom_labels=atom_type_labels,
                mask_indices=mask,  # Pass mask indices
                batch_idx=batch_idx
            )
            # divide by 2
            mask_loss = mask_loss / 2
        else:
            logger.debug("No mask loss computed")

        noise_loss = None
        if noise_mask is not None and denoise_vectors is not None:
            noise_loss = self.compute_noise_loss(
                output_vectors=output_vectors,
                target_vectors=denoise_vectors,
                loss_indices=noise_mask,
                batch_idx=batch_idx
            )
        else:
            logger.debug("No noise loss computed")

        return MetalSitePretrainingOutput(
            atom_logits=atom_logits,
            type_logits=type_logits,
            output_vectors=output_vectors,
            edge_attr=backbone_out.edge_attr,
            edge_embedding=backbone_out.edge_embedding,
            node_attr=backbone_out.node_attr,
            attentions=backbone_out.attentions,
            hidden_states=backbone_out.hidden_states,
            mask_loss=mask_loss,
            noise_loss=noise_loss
        )
    
    def embed_systems(
            self,
            atoms,
            atom_types,
            pos,
            batch_idx,
            tokenizer,
            atom_labels: Optional[torch.Tensor] = None,
            hidden_state: int = -1,
            how: str='<METAL>_mean', **kwargs):
        """Embed atomic data for a set of systems.
        
        Args:
            atoms: [num_atoms] Atom type tokens
            atom_types: [num_atoms] Record type tokens
            pos: [num_atoms, 3] Node coordinates
            batch_idx: [num_atoms] Batch assignments
            tokenizer: AtomTokenizer
                Tokenizer for embeddings - needed to de-tokenize and see which tokens to use
            how: str = '<METAL>_mean'
                Aggregation method for system embeddings
                First string is 'all' or a particular token to consider for embeddings
                Second string an aggregation method eg. 'mean', 'sum', 'max'
            atom_labels: Optional [num_atoms] True atom tokens. Used for indexing if available
            hidden_state: int = -1
                Hidden state index to use for embeddings
        Returns:
            MetalSiteFoundationalOutput containing node features and attributes
        """
        # set such that hidden states are returned
        self.backbone.backbone.output_hidden_states = True

        # Forward through backbone
        outputs = self.forward(
            atoms=atoms,
            atom_types=atom_types,
            pos=pos,
            batch_idx=batch_idx,
            **kwargs
        )

        # Extract hidden states
        hidden_states = outputs.hidden_states[hidden_state]
        # get only scalar embeddings
        # the others are higher order tensors
        is_l_zero = []
        for mul, ir in self.backbone.backbone.irreps_node_output:
            if ir.l == 0:
                is_l_zero.extend([True] * ir.dim * mul)
            else:
                is_l_zero.extend([False] * ir.dim * mul)
        embedding_indexes = torch.tensor(is_l_zero)

        # Get embeddings for atoms and atom types
        hidden_states = hidden_states[:, embedding_indexes]
        if atom_labels is not None:
            tokens_to_index = atom_labels
        else:
            tokens_to_index = atoms

        # de tokenize the embeddings
        token_strings = tokenizer.decode(atoms=tokens_to_index)['atoms']
        # convert to array if not already
        token_strings = np.array(token_strings)
        # get the embeddings for the tokens
        token_to_target, mean_type = how.split('_')
        if token_to_target == 'all':
            pass
        else:
            # ensure token is in tokenizer
            assert token_to_target in list(tokenizer.atom_vocab.itos.values()), f"{token_to_target} not in tokenizer"
            # get embeddings for only the target token
            correct_token_mask = token_strings == token_to_target
            hidden_states = hidden_states[correct_token_mask]
            batch_idx = batch_idx[correct_token_mask]

        # Aggregate embeddings
        # use multiple of present
        embeddings_to_cat = []
        if 'mean' in mean_type:
            embeddings_to_cat.append(scatter(hidden_states, batch_idx, dim=0, reduce='mean'))
        if 'sum' in mean_type:
            embeddings_to_cat.append(scatter(hidden_states, batch_idx, dim=0, reduce='sum'))
        if 'max' in mean_type:
            embeddings_to_cat.append(scatter(hidden_states, batch_idx, dim=0, reduce='max'))
        if 'min' in mean_type:
            embeddings_to_cat.append(scatter(hidden_states, batch_idx, dim=0, reduce='min'))
        embeddings = torch.cat(embeddings_to_cat, dim=-1)
        
        self.backbone.backbone.output_hidden_states = False
        
        return embeddings

                
