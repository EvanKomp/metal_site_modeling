"""
Null model evaluation script using training metrics.
Imported from metalsitenn.training.pretraining_scoring
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import dvc.api

from metalsitenn.utils import ParamsObj
from metalsitenn.dataloading import MetalSiteDataset
from metalsitenn.featurizer import MetalSiteCollator
from metalsitenn.training.pretraining_scoring import custom_eval_batch, f1_score_from_cm
from metalsitenn.graph_data import ModelOutput


def create_null_model_output(
    batch,
    most_frequent_class: int,
    vocab_size: int,
    class_weights: torch.Tensor,
    label_smoothing: float
) -> ModelOutput:
    """
    Create null model output that always predicts most frequent class.
    Uses exact same loss computation as the real model.
    
    Args:
        batch: Input batch
        most_frequent_class: Token ID of most frequent class
        vocab_size: Vocabulary size
        class_weights: Class weights for loss computation
        label_smoothing: Label smoothing factor
        
    Returns:
        ModelOutput with null predictions
    """
    batch_size = batch.element_labels.size(0)
    
    # Create logits strongly favoring most frequent class
    null_logits = torch.full(
        (batch_size, vocab_size), 
        -0.1, 
        dtype=torch.float32,
        device=batch.element_labels.device
    )
    null_logits[:, most_frequent_class] = 0.1
    
    # Compute cross-entropy loss EXACTLY like the real model
    # From model.py: cel_losses = self.cel(logits, batch.element_labels.squeeze(dim=1))
    cel = torch.nn.CrossEntropyLoss(
        reduction='none',  # Important: use 'none' like the real model
        weight=class_weights.to(batch.element_labels.device) if class_weights is not None else None,
        label_smoothing=label_smoothing
    )
    
    # Squeeze labels like the real model does
    labels_squeezed = batch.element_labels.squeeze(dim=1)  # N
    
    # Compute per-token losses: logits is (N, vocab_size), labels is (N)
    logits_2d = null_logits.view(-1, vocab_size)  # (N, vocab_size) 
    cel_losses = cel(logits_2d, labels_squeezed.view(-1))  # (N,)
    cel_losses = cel_losses.view(batch_size,1)
    
    # Apply masking exactly like the real model
    if batch.atom_masked_mask is not None:
        cel_losses = cel_losses * batch.atom_masked_mask.squeeze(-1)
        cel_loss = cel_losses.sum() / batch.atom_masked_mask.sum()
    else:
        cel_loss = cel_losses.mean()
    
    # Total loss = cel_loss (no auxiliary losses for null model)
    total_loss = cel_loss
    
    # Reshape logits back to (batch_size, num_nodes, vocab_size) for eval
    # This is critical - custom_eval_batch expects 3D logits
    final_logits = null_logits
    
    # Create ModelOutput
    model_output = ModelOutput(
        node_logits=final_logits,
        node_loss=cel_loss,
        loss=total_loss
    )
    
    return model_output


def compute_null_model_metrics(
    dataset: Subset,
    collator: MetalSiteCollator,
    class_weights: torch.Tensor,
    label_smoothing: float,
    split_name: str,
    logger: logging.Logger = None
) -> Dict[str, float]:
    """
    Compute null model metrics using same evaluation pipeline as training.
    
    Args:
        dataset: Dataset split to evaluate
        collator: MetalSiteCollator instance
        class_weights: Class weights tensor for loss computation
        label_smoothing: Label smoothing factor
        split_name: Name of dataset split for logging
        logger: Logger instance
        
    Returns:
        Dictionary with null model metrics matching training format
    """
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Find most frequent class from element counts
    element_counts_path = Path('data/2/2.0/element_counts.json')
    with open(element_counts_path, 'r') as f:
        element_counts_data = json.load(f)
    
    # Get token counts and find most frequent
    token_counts_dict = element_counts_data['element_tokens']
    token_counts = {}
    for token_id_str, count in token_counts_dict.items():
        token_id = int(token_id_str)
        token_counts[token_id] = count
    
    # Most frequent class (excluding potential padding/special tokens)
    most_frequent_class = max(token_counts.keys(), key=lambda x: token_counts[x])
    most_frequent_count = token_counts[most_frequent_class]
    
    if logger:
        logger.info(f"Most frequent class: {most_frequent_class} (count: {most_frequent_count})")
    
    vocab_size = len(collator.featurizer.tokenizers['element'].get_vocab())
    
    if logger:
        logger.info(f"Computing null model metrics for {split_name} split...")
    
    # Collect metrics using same evaluation pipeline as training
    all_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {split_name}"):
            # Create null model output
            model_output = create_null_model_output(
                batch=batch,
                most_frequent_class=most_frequent_class,
                vocab_size=vocab_size,
                class_weights=class_weights,
                label_smoothing=label_smoothing
            )
            
            # Debug: Check ranges before calling custom_eval_batch
            if batch.atom_masked_mask is not None:
                mask = batch.atom_masked_mask.squeeze(-1)
                labels_masked = batch.element_labels[mask]
                logits_masked = model_output.node_logits[mask]
                preds_masked = logits_masked.argmax(dim=-1)
                
                # Validate ranges
                if labels_masked.min() < 0 or labels_masked.max() >= vocab_size:
                    if logger:
                        logger.error(f"Invalid label range: [{labels_masked.min().item()}, {labels_masked.max().item()}], vocab_size: {vocab_size}")
                    continue
                    
                if preds_masked.min() < 0 or preds_masked.max() >= vocab_size:
                    if logger:
                        logger.error(f"Invalid pred range: [{preds_masked.min().item()}, {preds_masked.max().item()}], vocab_size: {vocab_size}")
                    continue
            
            # Use same evaluation function as training
            batch_metrics = custom_eval_batch(batch, model_output)
            
            # Accumulate metrics (same as training evaluation)
            for key, tensor in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(tensor.cpu())
    
    # Compute final metrics using same aggregation as training
    cat_metrics = {}
    for key, value in all_metrics.items():
        if len(value) == 0:
            continue
        # Handle scalar tensors by stacking instead of concatenating
        first_tensor = value[0]
        if first_tensor.dim() == 0:  # Scalar tensor
            cat_metrics[key] = torch.stack(value, dim=0)
        else:  # Multi-dimensional tensor
            cat_metrics[key] = torch.cat(value, dim=0)
    
    # Compute aggregated metrics (same as custom_eval_logger)
    out_metrics = {}
    
    # Mean losses
    out_metrics['loss'] = cat_metrics['loss'].float().mean().item() 
    out_metrics['cel_loss'] = cat_metrics['cel_loss'].float().mean().item()

    # Accuracy from prediction counts
    total_correct = cat_metrics['correct_preds'].sum().item()
    total_incorrect = cat_metrics['incorrect_preds'].sum().item()
    out_metrics['accuracy'] = total_correct / (total_correct + total_incorrect) 

    # Metal accuracy from confusion matrix
    cm = cat_metrics['cm'].sum(dim=0)  # C, C
    metal_token_index = collator.featurizer.tokenizers['element'].metal_token_id
    metal_cm_row = cm[metal_token_index]
    metal_accuracy = metal_cm_row[metal_token_index].item() / metal_cm_row.sum().item()
    out_metrics['metal_accuracy'] = metal_accuracy

    # F1 score
    f1 = f1_score_from_cm(cm.numpy())
    out_metrics['f1'] = f1
    
    # Add null model specific info
    out_metrics['most_frequent_class'] = most_frequent_class
    out_metrics['most_frequent_count'] = most_frequent_count
    
    if logger:
        logger.info(f"{split_name} null model metrics:")
        logger.info(f"  Loss: {out_metrics['loss']:.6f}")
        logger.info(f"  CEL Loss: {out_metrics['cel_loss']:.6f}")
        logger.info(f"  Accuracy: {out_metrics['accuracy']:.6f}")
        logger.info(f"  Metal Accuracy: {out_metrics['metal_accuracy']:.6f}")
        logger.info(f"  F1 Score: {out_metrics['f1']:.6f}")
    
    return out_metrics


def evaluate_null_model_splits(
    val_dataset: Subset,
    test_dataset: Subset,
    collator: MetalSiteCollator,
    class_weights: torch.Tensor,
    label_smoothing: float,
    logger: logging.Logger = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate null model on validation and test splits using training metrics.
    
    Args:
        val_dataset: Validation dataset
        test_dataset: Test dataset  
        collator: MetalSiteCollator instance
        class_weights: Class weights tensor
        label_smoothing: Label smoothing factor
        logger: Logger instance
        
    Returns:
        Dictionary with metrics for val and test splits
    """
    all_metrics = {}
    
    for split_name, dataset_split in [
        ('val', val_dataset),
        ('test', test_dataset)
    ]:
        if logger:
            logger.info(f"\nProcessing {split_name} split...")
        
        split_metrics = compute_null_model_metrics(
            dataset=dataset_split,
            collator=collator,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            split_name=split_name,
            logger=logger
        )
        
        all_metrics[split_name] = split_metrics
    
    return all_metrics


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


def load_dvc_params() -> ParamsObj:
    """Load DVC parameters for 2_pretraining stage."""
    params = dvc.api.params_show()
    return ParamsObj(params)['_2_pretraining']


def initialize_collator(params: ParamsObj) -> MetalSiteCollator:
    """Initialize MetalSiteCollator from DVC params."""
    tokenization_params = params.data.tokenization
    
    collator = MetalSiteCollator(
        **tokenization_params
    )
    
    return collator


def load_and_split_dataset(params: ParamsObj) -> tuple[Subset, Subset, Subset]:
    """Load dataset and create train/validation/test splits."""
    filtering_params = params.data.filtering
    
    # Initialize dataset with filtering parameters
    cache_folder = 'data/1/1.1_parse_sites_metadata'
    dataset = MetalSiteDataset(
        cache_folder=cache_folder,
        overwrite=False,
        save_pdb=False,
        **filtering_params
    )
    
    logging.info(f"Loaded dataset with {len(dataset)} sites after filtering")
    
    # Create deterministic train/val/test split
    split_params = params.data.splitting
    test_frac = split_params.test_frac
    val_frac = split_params.val_frac_of_train
    
    dataset_size = len(dataset)
    train_size = round((1 - test_frac - val_frac) * dataset_size)
    val_size = round(val_frac * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Set seed for reproducible splits
    split_seed = split_params.seed
    np.random.seed(split_seed)
    
    # Generate random indices
    indices = np.random.permutation(dataset_size)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # if debug size
    if params.data.debug_max_sites:
        if len(train_indices) > params.data.debug_max_sites:
            train_indices = train_indices[:params.data.debug_max_sites]
        if len(val_indices) > params.data.debug_max_sites:
            val_indices = val_indices[:params.data.debug_max_sites]
        if len(test_indices) > params.data.debug_max_sites:
            test_indices = test_indices[:params.data.debug_max_sites]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    logging.info(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main function to compute null model metrics using training evaluation pipeline."""
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/2.2_compute_null_model_training_metrics.log')
    
    logger.info("Starting null model evaluation with training metrics")
    
    # Load DVC parameters
    logger.info("Loading DVC parameters...")
    params = load_dvc_params()
    
    # Initialize collator
    logger.info("Initializing collator...")
    collator = initialize_collator(params)
    logger.info(f"Collator initialized with vocab sizes: {collator.featurizer.get_feature_vocab_sizes()}")
    
    # Load class weights if specified
    class_weights = None
    label_smoothing = params.model.node_class_label_smoothing
    
    if params.model.node_class_weights:
        weights_path = Path('data/2/2.1/node_class_weights.json')
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
            class_weights = torch.tensor(weights_data['class_weights'], dtype=torch.float32)
            logger.info(f"Loaded class weights from {weights_path}")
        else:
            logger.warning(f"Class weights file not found: {weights_path}")
    
    # Load and split dataset
    logger.info("Loading and splitting dataset...")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(params)
    
    # Create output directory
    output_dir = Path('data/metrics/2/2.2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate null model on val and test splits only
    logger.info("Computing null model metrics...")
    all_metrics = evaluate_null_model_splits(
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        collator=collator,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        logger=logger
    )
    
    # Save results
    output_path = output_dir / 'null_model_training_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"\nSaved null model metrics to {output_path}")
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("NULL MODEL TRAINING METRICS SUMMARY")
    logger.info("="*60)
    
    for split_name in ['val', 'test']:
        metrics = all_metrics[split_name]
        logger.info(f"{split_name.upper()} SET:")
        logger.info(f"  Loss: {metrics['loss']:.6f}")
        logger.info(f"  CEL Loss: {metrics['cel_loss']:.6f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.6f}")
        logger.info(f"  Metal Accuracy: {metrics['metal_accuracy']:.6f}")
        logger.info(f"  F1 Score: {metrics['f1']:.6f}")
        logger.info(f"  Most frequent class: {metrics['most_frequent_class']}")
        
    logger.info("\nNull model evaluation completed successfully!")


if __name__ == "__main__":
    main()