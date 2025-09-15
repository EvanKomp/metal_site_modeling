# pipeline/2_pretraining/2.1_compute_node_class_weights.py
'''
* Author: Evan Komp
* Created: 8/21/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import dvc.api

from metalsitenn.utils import ParamsObj
from metalsitenn.tokenizers import TOKENIZERS

# Use non-interactive backend for headless environments
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_weights_plot(class_weights: torch.Tensor, 
                       id_to_token: Dict[int, str],
                       token_counts: torch.Tensor,
                       temperature: float,
                       output_path: Path,
                       clip_token: Optional[str] = None,
                       clip_weight: Optional[float] = None) -> None:
    """
    Create a plot of class weights ordered by weight value with dual y-axis for frequency.
    
    Args:
        class_weights: Tensor of class weights
        id_to_token: Mapping from token IDs to strings
        token_counts: Tensor of token counts parallel to class_weights
        temperature: Temperature parameter used
        output_path: Path to save the plot
        clip_token: Token used for clipping (for annotation)
        clip_weight: Weight value used for clipping (for annotation)
    """
    # Get non-zero weights and their corresponding info
    nonzero_mask = class_weights > 0
    nonzero_weights = class_weights[nonzero_mask]
    nonzero_indices = torch.where(nonzero_mask)[0]
    nonzero_counts = token_counts[nonzero_mask]
    
    # Sort by weight (descending)
    sorted_indices = torch.argsort(nonzero_weights, descending=True)
    sorted_weights = nonzero_weights[sorted_indices]
    sorted_token_indices = nonzero_indices[sorted_indices]
    sorted_counts = nonzero_counts[sorted_indices]
    
    # Get token names for sorted weights
    token_names = []
    for idx in sorted_token_indices:
        token_id = idx.item()
        token_name = id_to_token.get(token_id, f"UNK_{token_id}")
        token_names.append(token_name)
    
    # Create the plot with dual y-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Weights ordered by value with frequency on dual y-axis
    x_pos = np.arange(len(sorted_weights))
    
    # Primary y-axis: weights
    bars1 = ax1.bar(x_pos, sorted_weights.numpy(), alpha=0.7, color='steelblue', label='Class Weight')
    ax1.set_xlabel('Token Index (ordered by weight)')
    ax1.set_ylabel('Class Weight', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(True, alpha=0.3)
    
    # Secondary y-axis: frequency (log scale)
    ax1_freq = ax1.twinx()
    line1 = ax1_freq.plot(x_pos, sorted_counts.numpy(), 'ro-', alpha=0.6, 
                         markersize=3, linewidth=1, label='Token Count')
    ax1_freq.set_ylabel('Token Count (log scale)', color='red')
    ax1_freq.set_yscale('log')
    ax1_freq.tick_params(axis='y', labelcolor='red')
    
    # Title with clipping info
    title = f'Node Class Weights (Temperature = {temperature})\nOrdered by Weight Value'
    if clip_token and clip_weight is not None:
        title += f'\nClipped to {clip_token} weight: {clip_weight:.4f}'
    ax1.set_title(title)
    
    # Add clipping line if applicable
    if clip_weight is not None:
        ax1.axhline(y=clip_weight, color='orange', linestyle='--', alpha=0.8, 
                   label=f'Clip at {clip_weight:.4f}')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_freq.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Annotate all tokens
    for i in range(len(token_names)):
        # Highlight clip token differently
        if clip_token and token_names[i] == clip_token:
            ax1.annotate(token_names[i], 
                        (i, sorted_weights[i]), 
                        rotation=45, 
                        ha='left', 
                        va='bottom',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="orange", alpha=0.7))
        else:
            ax1.annotate(token_names[i], 
                        (i, sorted_weights[i]), 
                        rotation=45, 
                        ha='left', 
                        va='bottom',
                        fontsize=8)
    
    # Plot 2: Counts vs Weights scatter (both log scale)
    ax2.scatter(sorted_counts.numpy(), sorted_weights.numpy(), alpha=0.6, color='darkred')
    ax2.set_xlabel('Token Count (log scale)')
    ax2.set_ylabel('Class Weight (log scale)')
    ax2.set_title('Class Weight vs Token Count')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add clipping line if applicable
    if clip_weight is not None:
        ax2.axhline(y=clip_weight, color='orange', linestyle='--', alpha=0.8, 
                   label=f'Clip at {clip_weight:.4f}')
        ax2.legend()
    
    # Annotate all points
    for i in range(len(token_names)):
        # Always annotate clip token if present with special highlighting
        if clip_token and token_names[i] == clip_token:
            ax2.annotate(f'{token_names[i]} (clip target)', 
                        (sorted_counts[i], sorted_weights[i]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        else:
            ax2.annotate(token_names[i], 
                        (sorted_counts[i], sorted_weights[i]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_class_weights_with_temperature(token_counts: torch.Tensor, 
                                         temperature: float,
                                         clip_token_id: Optional[int] = None) -> torch.Tensor:
    """
    Compute class weights using temperature-based smoothing with optional clipping.
    
    Args:
        token_counts: Tensor of shape [vocab_size] with token counts
        temperature: Temperature parameter for smoothing (higher = more uniform)
        clip_token_id: Optional token ID to use as clipping threshold
        
    Returns:
        Tensor of shape [vocab_size] with class weights
    """
    # Initialize weights tensor with zeros
    weights = torch.zeros_like(token_counts, dtype=torch.float32)
    
    # Find tokens with non-zero counts
    nonzero_mask = token_counts > 0
    nonzero_counts = token_counts[nonzero_mask]
    
    if nonzero_counts.numel() == 0:
        logger.warning("No valid counts found, returning uniform weights")
        return weights
    
    # Apply temperature smoothing: weight ∝ (1/count)^(1/temperature)
    # Higher temperature makes weights more uniform
    # Lower temperature makes rare classes have much higher weights
    if temperature > 0:
        # Inverse frequency with temperature
        inv_freq = 1.0 / nonzero_counts.float()
        weights_unnormalized = torch.pow(inv_freq, 1.0 / temperature)
    else:
        # Temperature = 0 means uniform weights for non-zero classes
        weights_unnormalized = torch.ones_like(nonzero_counts, dtype=torch.float32)
    
    # Apply clipping if clip_token_id is provided
    clip_weight = None
    if clip_token_id is not None and clip_token_id < len(token_counts):
        # Find the clip token in the nonzero tokens
        clip_token_position = None
        nonzero_token_ids = torch.where(nonzero_mask)[0]
        
        for i, token_id in enumerate(nonzero_token_ids):
            if token_id == clip_token_id:
                clip_token_position = i
                break
        
        if clip_token_position is not None:
            clip_weight = weights_unnormalized[clip_token_position].item()
            logger.info(f"Clipping weights to clip token weight: {clip_weight:.6f}")
            
            # Clip all weights to not exceed the clip token weight
            weights_unnormalized = torch.clamp(weights_unnormalized, max=clip_weight)
            
            # Count how many weights were clipped
            original_max = torch.pow(1.0 / nonzero_counts.float().min(), 1.0 / temperature)
            n_clipped = (weights_unnormalized == clip_weight).sum().item()
            logger.info(f"Clipped {n_clipped} weights from max {original_max:.6f} to {clip_weight:.6f}")
        else:
            logger.warning(f"Clip token ID {clip_token_id} not found in non-zero tokens, no clipping applied")
    
    # Normalize so that total weight equals number of non-zero classes
    # This maintains the same "total importance" regardless of distribution
    total_weight = nonzero_mask.sum().float()
    weights_normalized = weights_unnormalized * total_weight / weights_unnormalized.sum()
    
    # Assign weights to corresponding indices
    weights[nonzero_mask] = weights_normalized
    
    return weights, clip_weight

def main():
    """Main function to compute node class weights."""
    # Load parameters
    PARAMS = dvc.api.params_show()
    PARAMS = ParamsObj(PARAMS)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/2.1_compute_node_class_weights.log')
    
    logger.info("Starting node class weights computation")
    
    # Extract parameters
    active_aggregators = PARAMS['2_pretraining']['data']['tokenization']['active_aggregators']
    temperature = PARAMS['2_pretraining']['data']['loss_weighting']['cel_count_to_weight_temperature']
    clip_token = PARAMS['2_pretraining']['data']['loss_weighting']['cel_count_to_weight_clip_token']
    
    logger.info(f"Active aggregators: {active_aggregators}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Clip token: {clip_token}")
    
    # Create output directory
    output_dir = Path('data/2/2.1')
    plots_dir = Path('data/plots/2/2.1')
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load element counts
    counts_path = Path('data/2/2.0/element_counts.json')
    if not counts_path.exists():
        raise FileNotFoundError(f"Element counts file not found: {counts_path}")
    
    with open(counts_path, 'r') as f:
        element_counts_data = json.load(f)
    
    logger.info(f"Loaded element counts from {counts_path}")
    
    # Get element tokenizer to determine vocabulary
    element_tokenizer = TOKENIZERS['element']
    vocab_size = element_tokenizer.vocab_size
    
    logger.info(f"Element vocabulary size: {vocab_size}")
    
    # Use token counts (which are now int to int)
    element_token_counts_dict = element_counts_data['element_tokens']
    
    # Convert string keys back to int if they were saved as strings in JSON
    token_counts_dict = {}
    for token_id_str, count in element_token_counts_dict.items():
        token_id = int(token_id_str)  # Convert string key back to int
        token_counts_dict[token_id] = int(count)
    
    # Create count tensor from dict
    token_counts = torch.zeros(vocab_size, dtype=torch.long)
    for token_id, count in token_counts_dict.items():
        if 0 <= token_id < vocab_size:
            token_counts[token_id] = count
        else:
            logger.warning(f"Token ID {token_id} is out of vocabulary range [0, {vocab_size})")
    
    logger.info(f"Processing {len(token_counts_dict)} token types")
    
    # Count non-zero classes
    non_zero_classes = (token_counts > 0).sum().item()
    total_tokens = token_counts.sum().item()
    
    logger.info(f"Non-zero classes: {non_zero_classes}/{vocab_size}")
    logger.info(f"Total tokens: {total_tokens:,}")
    
    # Get clip token ID if specified
    clip_token_id = None
    if clip_token is not None:
        clip_token_id = element_tokenizer.d2i.get(clip_token, None)
        if clip_token_id is None:
            logger.warning(f"Clip token '{clip_token}' not found in vocabulary, no clipping will be applied")
        else:
            logger.info(f"Using clip token '{clip_token}' (ID: {clip_token_id})")
    
    # Compute class weights
    class_weights, clip_weight = compute_class_weights_with_temperature(
        token_counts=token_counts,
        temperature=temperature,
        clip_token_id=clip_token_id
    )
    
    # Validate weights
    non_zero_weights = (class_weights > 0).sum().item()
    total_weight = class_weights.sum().item()
    
    logger.info(f"Computed weights: {non_zero_weights} non-zero weights")
    logger.info(f"Total weight: {total_weight:.2f} (target: {non_zero_classes})")
    
    # Log some statistics about the weights
    if non_zero_weights > 0:
        nonzero_weights = class_weights[class_weights > 0]
        logger.info(f"Weight statistics:")
        logger.info(f"  Min weight: {nonzero_weights.min().item():.4f}")
        logger.info(f"  Max weight: {nonzero_weights.max().item():.4f}")
        logger.info(f"  Mean weight: {nonzero_weights.mean().item():.4f}")
        logger.info(f"  Std weight: {nonzero_weights.std().item():.4f}")
        if clip_weight is not None:
            logger.info(f"  Clip weight: {clip_weight:.4f}")
    
    # Show top weighted classes (rarest tokens)
    top_indices = torch.topk(class_weights, k=min(10, non_zero_weights)).indices
    logger.info("Top 10 weighted classes (rarest tokens):")
    id_to_token = {v: k for k, v in element_tokenizer.d2i.items()}
    for i, idx in enumerate(top_indices):
        token_id = idx.item()
        weight = class_weights[token_id].item()
        token_str = id_to_token.get(token_id, f"UNK_{token_id}")
        count = token_counts[token_id].item()
        clipped_indicator = " (CLIPPED)" if clip_weight is not None and abs(weight - clip_weight) < 1e-6 else ""
        logger.info(f"  {i+1:2d}. {token_str:12s}: weight={weight:.4f}, count={count}{clipped_indicator}")
    
    # Prepare output data
    output_data = {
        'class_weights': class_weights.tolist(),
        'vocab_size': vocab_size,
        'temperature': temperature,
        'clip_token': clip_token,
        'clip_weight': float(clip_weight) if clip_weight is not None else None,
        'non_zero_classes': non_zero_classes,
        'total_weight': total_weight,
        'active_aggregators': active_aggregators,
        'metadata': {
            'total_tokens_processed': total_tokens,
            'weight_statistics': {
                'min_weight': float(nonzero_weights.min()) if non_zero_weights > 0 else 0.0,
                'max_weight': float(nonzero_weights.max()) if non_zero_weights > 0 else 0.0,
                'mean_weight': float(nonzero_weights.mean()) if non_zero_weights > 0 else 0.0,
                'std_weight': float(nonzero_weights.std()) if non_zero_weights > 0 else 0.0,
            }
        }
    }
    
    # Save class weights
    output_path = output_dir / 'node_class_weights.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved node class weights to {output_path}")
    
    # Create visualization plot
    plot_path = plots_dir / 'node_class_weights.png'
    try:
        create_weights_plot(
            class_weights=class_weights,
            id_to_token=id_to_token,
            token_counts=token_counts,
            temperature=temperature,
            output_path=plot_path,
            clip_token=clip_token,
            clip_weight=clip_weight
        )
        logger.info(f"Saved weights plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")
        # Don't fail the entire script if plotting fails
    
    # Log summary
    logger.info("=" * 50)
    logger.info("CLASS WEIGHTS COMPUTATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Clip token: {clip_token}")
    if clip_weight is not None:
        logger.info(f"Clip weight: {clip_weight:.6f}")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Classes with data: {non_zero_classes}")
    logger.info(f"Total weight assigned: {total_weight:.2f}")
    logger.info(f"Weight normalization: target={non_zero_classes}, actual={total_weight:.2f}")
    
    if temperature == 0:
        logger.info("Temperature=0: Uniform weights for non-zero classes")
    elif temperature < 1:
        logger.info(f"Temperature={temperature}: Strong emphasis on rare classes")
    elif temperature == 1:
        logger.info("Temperature=1: Inverse frequency weighting")
    else:
        logger.info(f"Temperature={temperature}: Gentle emphasis on rare classes")
    
    if clip_token and clip_weight is not None:
        logger.info(f"Applied clipping: weights capped at {clip_token} weight ({clip_weight:.6f})")
    
    logger.info("Node class weights computation completed successfully!")

if __name__ == "__main__":
    main()