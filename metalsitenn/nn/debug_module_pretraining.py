# metalsitenn/nn/debug_module_pretraining.py
'''
* Author: Evan Komp
* Created: 8/29/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import torch
import torch.nn as nn
from typing import Dict, Any
from metalsitenn.nn.mlp import MLP
from metalsitenn.graph_data import ModelOutput


class SimpleDebugModel(nn.Module):
    """
    Simple debug model for trainer testing.
    
    Expects N,1 tokens and N,1 labels, embeds them, processes through MLPs,
    outputs vocab_size logits and cross-entropy loss.
    
    Args:
        vocab_size: Size of vocabulary for embedding and output logits
        embed_dim: Embedding dimension (default: 64)
        hidden_dim: Hidden layer dimension (default: 128)
        n_layers: Number of MLP layers (default: 2)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        cel_class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # MLP layers using the existing MLP module
        self.mlp = MLP(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            n_hidden_layers=n_layers,
            hidden_activation='relu',
            dropout_rate=0.0
        )
        
        # Output projection to vocab size
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Loss function
        self.cel = nn.CrossEntropyLoss(reduction='mean', weight=cel_class_weights, label_smoothing=label_smoothing)

    def forward(
        self, 
        batch: Any,
        compute_loss: bool = True,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through debug model - pure overfitting experiment.
        Embeds the labels and tries to predict them back.
        
        Args:
            batch: Batch object with .element_labels (N,1 labels)
            compute_loss: Whether to compute loss
            **kwargs: Additional arguments (ignored)
            
        Returns:
            ModelOutput with node_logits and optionally loss/node_loss
        """
        # Extract labels as input tokens - expect (N, 1) shape, squeeze to (N,)
        tokens = batch.element_labels.squeeze(-1) if batch.element_labels.dim() > 1 else batch.element_labels
        
        # Embedding: (N,) -> (N, embed_dim)
        embeddings = self.embedding(tokens)
        
        # MLP processing: (N, embed_dim) -> (N, hidden_dim)
        hidden = self.mlp(embeddings)
        
        # Output projection: (N, hidden_dim) -> (N, vocab_size)  
        logits = self.output_proj(hidden)
        
        # Prepare output - always include logits
        output_data = {'node_logits': logits}
        
        if compute_loss:
            # Labels are same as input tokens for pure overfitting
            labels = tokens
            
            # Compute loss on all tokens (no masking for debug)
            # convert labels to float
            node_loss = self.cel(logits, labels.long())
            total_loss = node_loss
            
            output_data.update({
                'loss': total_loss,
                'node_loss': node_loss
            })
        
        return ModelOutput(**output_data)