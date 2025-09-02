# pipeline/2_pretraining/2.4_evaluate.py
'''
* Author: Evan Komp
* Created: 9/2/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import dvc.api
from tqdm import tqdm

from metalsitenn.utils import ParamsObj
from metalsitenn.dataloading import MetalSiteDataset
from metalsitenn.featurizer import MetalSiteCollator
# No need to import config - from_pretrained handles it
from metalsitenn.nn.model import EquiformerWEdgesForPretraining
from metalsitenn.training.pretraining_scoring import custom_eval_batch, f1_score_from_cm
from metalsitenn.plotting import confusion_from_matrix

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
    return ParamsObj(params)['2_pretraining']

def initialize_collator(params: ParamsObj) -> MetalSiteCollator:
    """Initialize collator from DVC params."""
    tokenization_params = params.data.tokenization
    
    collator = MetalSiteCollator(
        **tokenization_params
    )
    
    return collator

# Model config not needed - from_pretrained handles this automatically

def load_and_split_dataset(params: ParamsObj) -> tuple[Subset, Subset, Subset]:
    """Load dataset and create train/validation/test splits."""
    filtering_params = params.data.filtering
    
    # Initialize dataset with filtering parameters
    cache_folder = '../bonnanzio_metal_site_modeling/data/1/1.1_parse_sites_metadata'
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

def compute_final_metrics(metrics: Dict[str, torch.Tensor], collator: MetalSiteCollator) -> Dict[str, float]:
    """
    Compute final evaluation metrics from concatenated batch results.
    Mirrors custom_eval_logger without trainer dependency or logging.
    """
    # Concatenate all metrics along first dimension
    cat_metrics = {}
    for key, value in metrics.items():
        if len(value) == 0:
            continue
        # Handle scalar tensors by stacking instead of concatenating
        first_tensor = value[0]
        if first_tensor.dim() == 0:  # Scalar tensor
            cat_metrics[key] = torch.stack(value, dim=0)
        else:  # Multi-dimensional tensor
            cat_metrics[key] = torch.cat(value, dim=0)
    
    # Compute aggregated metrics
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
    
    return out_metrics

def evaluate_dataset(model: torch.nn.Module, 
                    dataloader: DataLoader, 
                    collator: MetalSiteCollator,
                    device: torch.device,
                    split_name: str) -> Dict[str, float]:
    """Evaluate model on a dataset split."""
    model.eval()
    
    all_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            model_output = model(batch)
            
            # Compute batch metrics
            batch_metrics = custom_eval_batch(batch, model_output)
            
            # Accumulate metrics
            for key, tensor in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(tensor.cpu())
    
    # Compute final metrics
    final_metrics = compute_final_metrics(all_metrics, collator)
    
    return final_metrics

def main():
    """Main evaluation function."""
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/2.4_evaluation.log')
    
    logger.info("Starting pretrained model evaluation")
    
    # Load DVC parameters
    logger.info("Loading DVC parameters...")
    params = load_dvc_params()
    logger.info("Loaded parameters for 2_pretraining stage")
    
    # Initialize components
    logger.info("Initializing collator...")
    collator = initialize_collator(params)
    logger.info(f"Collator initialized with vocab sizes: {collator.featurizer.get_feature_vocab_sizes()}")
    
    # Load dataset splits
    logger.info("Loading and splitting dataset...")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(params)
    logger.info("Dataset loaded and split successfully")
    
    # Load pretrained model
    logger.info("Loading pretrained model...")
    run_dir = params.training.run_dir
    model_path = Path(run_dir) / "final_model"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Final model not found at {model_path}. Make sure training has completed.")
    
    model = EquiformerWEdgesForPretraining.from_pretrained(model_path)
    logger.info("Pretrained model loaded successfully")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Create dataloaders
    batch_size = params.training.batch_size
    num_workers = params.training.num_workers
    pin_memory = params.training.pin_memory
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_dataset(model, val_loader, collator, device, "validation")
    logger.info("Validation metrics:")
    for key, value in val_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_dataset(model, test_loader, collator, device, "test")
    logger.info("Test metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    
    # Create metrics directory
    os.makedirs("data/metrics/2/2.4", exist_ok=True)

    # Save validation metrics
    with open("data/metrics/2/2.4/val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    # Save test metrics
    with open("data/metrics/2/2.4/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    
    logger.info("Metrics saved to metrics/ directory for DVC tracking")
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()