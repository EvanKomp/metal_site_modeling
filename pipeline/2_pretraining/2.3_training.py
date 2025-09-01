# pipeline/2_pretraining/2.3_training.py
'''
* Author: Evan Komp
* Created: 8/22/2025
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
from torch.utils.data import Subset
import dvc.api

from metalsitenn.utils import ParamsObj
from metalsitenn.dataloading import MetalSiteDataset
from metalsitenn.featurizer import MetalSiteCollator
from metalsitenn.nn.pretrained_config import EquiformerWEdgesConfig
from metalsitenn.training.trainer import TrainerConfig, MetalSiteTrainer
from metalsitenn.nn.model import EquiformerWEdgesForPretraining
from metalsitenn.plotting import confusion_from_matrix

######################################################################
# Custom eval functions related to node prediction task
######################################################################

def f1_score_from_cm(cm):
    """Calculate weighted F1 score directly from confusion matrix"""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    # Avoid division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1_per_class = np.where(precision + recall > 0, 
                               2 * precision * recall / (precision + recall), 0)
    
    support = cm.sum(axis=1)
    return (f1_per_class * support).sum() / support.sum()

def custom_eval_batch(batch: "BatchProteinData", model_outs: "ModelOuts"):
    """Compute metrics from outputs logits and loss.
    
    Metrics:
    - loss (easy)
    - cel_loss (eg. minus auxillary losses)
    - correct_preds, incorrect_preds (for accuracy)
    - cm, [1,C,C] confusion matrix of counts - we want to sum these later
    """
    out_metrics = {}
    cel_loss = model_outs.node_loss
    loss = model_outs.loss
    out_metrics['loss'] = loss.detach()
    out_metrics['cel_loss'] = cel_loss.detach()

    # labels and predictions
    atom_masked_mask = batch.atom_masked_mask.detach() # N, 1, sum = M
    labels = batch.element_labels.detach()[atom_masked_mask.squeeze(-1)] # M, 1
    logits = model_outs.node_logits.detach()[atom_masked_mask.squeeze(-1)] # M, num_classes
    preds = logits.argmax(dim=-1).unsqueeze(-1) # M, 1
    total_num_classes = logits.size(1)

    corrects_mask = (preds == labels).squeeze(-1) # M, 1
    incorrects_mask = (preds != labels).squeeze(-1) # M, 1
    out_metrics['correct_preds'] = corrects_mask.sum().unsqueeze(0)
    out_metrics['incorrect_preds'] = incorrects_mask.sum().unsqueeze(0)

    # Efficient confusion matrix computation using advanced indexing
    labels_flat = labels.squeeze(-1)  # M   
    preds_flat = preds.squeeze(-1)    # M

    # now build the counts in a confusion matrix
    cm_indices = labels_flat * total_num_classes + preds_flat  # M
    
    # Use bincount to efficiently count occurrences
    cm_counts = torch.bincount(cm_indices, minlength=total_num_classes * total_num_classes)
    
    # Reshape to confusion matrix format and convert to individual entries
    cm_matrix = cm_counts.view(total_num_classes, total_num_classes).unsqueeze(0)  # 1, C, C such that we can cat them later
    out_metrics['cm'] = cm_matrix
    
    return out_metrics

def custom_eval_logger(trainer, metrics: Dict[str, torch.Tensor]):
    """Custom evaluation logging function."""
    # we can cat them all along first dimension
    cat_metrics = {}
    for key, value in metrics.items():
        cat_metrics[key] = torch.cat(value, dim=0)
        trainer.log_debug(f"Shape of catted eval metric {key} after full val set: {cat_metrics[key].shape}")

    # Compute aggregated metrics
    # losses we just mean
    out_metrics = {}
    out_metrics['loss'] = cat_metrics['loss'].float().mean().item() 
    out_metrics['cel_loss'] = cat_metrics['cel_loss'].float().mean().item()

    # with the pred counts we can get accuracy
    total_correct = cat_metrics['correct_preds'].sum().item()
    total_incorrect = cat_metrics['incorrect_preds'].sum().item()
    out_metrics['accuracy'] = total_correct / (total_correct + total_incorrect) 

    # do metal accuracy
    # we can use the confusion matrix to do this
    cm = cat_metrics['cm'].sum(dim=0) # C, C
    metal_token_index = trainer.collator.featurizer.tokenizers['element'].metal_token_id
    out_metrics['metal_accuracy'] = cm[metal_token_index, metal_token_index].item() / cm[metal_token_index].sum().item()

    # f1 score
    f1 = f1_score_from_cm(cm.numpy())
    out_metrics['f1'] = f1

    # and now we can log the confusion matrix as a sklearn plot with dvc live
    fig = confusion_from_matrix(
        cm, vocab=tuple(trainer.collator.featurizer.tokenizers['element'].get_vocab().keys()),
        normalize=True
    )
    if trainer.accelerator.is_main_process:
        live = trainer.accelerator.get_tracker("dvclive", unwrap=True)
        live.log_image("eval_confusion_matrix.png", fig)

    # also log the metrics
    trainer._log_metrics(out_metrics, prefix='eval/')
    return out_metrics

######################################################################

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""

    # only for main process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:

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


def initialize_model_config(params: ParamsObj, collator: MetalSiteCollator) -> EquiformerWEdgesConfig:
    """Initialize model configuration from DVC params."""
    model_params = params.model.copy().dict()
    
    # Extract atom and bond features from collator params
    tokenization_params = params.data.tokenization
    atom_features = tokenization_params.atom_features
    bond_features = tokenization_params.bond_features
    
    # Get feature vocab sizes from collator
    feature_vocab_sizes = collator.featurizer.get_feature_vocab_sizes()

    # avg degree equals k since each subgraph is always k
    model_params['avg_degree'] = tokenization_params.k

    # avg_num_nodes
    with open('data/metrics/2/2.0/data_stats.json', 'r') as f:
        dataset_stats = json.load(f)
        avg_num_nodes = dataset_stats['avg_atoms_per_site']
    model_params['avg_num_nodes'] = avg_num_nodes
    
    # Load node class weights if specified
    if model_params['node_class_weights'] is not None:
        if isinstance(model_params['node_class_weights'], bool) and model_params['node_class_weights']:
            # Load from computed weights file
            weights_path = Path('data/2/2.1/node_class_weights.json')
            if weights_path.exists():
                with open(weights_path, 'r') as f:
                    weights_data = json.load(f)
                model_params['node_class_weights'] = torch.tensor(weights_data['class_weights'])
                logging.info(f"Loaded node class weights from {weights_path}")
            else:
                logging.warning(f"Node class weights file not found: {weights_path}")
                model_params['node_class_weights'] = None
        # If it's already a list/array, keep as is
    
    model_config = EquiformerWEdgesConfig(
        atom_features=atom_features,
        bond_features=bond_features,
        feature_vocab_sizes=feature_vocab_sizes,
        **model_params
    )
    
    return model_config


def initialize_collator(params: ParamsObj) -> MetalSiteCollator:
    """Initialize MetalSiteCollator from DVC params."""
    tokenization_params = params.data.tokenization
    
    collator = MetalSiteCollator(
        **tokenization_params
    )
    
    return collator


def initialize_training_config(params: ParamsObj) -> TrainerConfig:
    """Initialize training configuration from DVC params."""
    training_params = params.training
    
    # Build training config - all parameters must be explicitly set in params.yaml
    training_config = TrainerConfig(
        # Core training parameters
        max_epochs=training_params.max_epochs,
        eval_every=training_params.eval_every,
        log_every=training_params.log_every,
        max_checkpoints=training_params.max_checkpoints,
        save_best_only=training_params.save_best_only,
        
        # Learning rate & scheduling
        lr_initial=training_params.lr_initial,
        scheduler=training_params.scheduler,
        lambda_type=training_params.lambda_type,
        warmup_epochs=training_params.warmup_epochs,
        warmup_factor=training_params.warmup_factor,
        lr_min_factor=training_params.lr_min_factor,
        decay_epochs=training_params.decay_epochs,
        decay_rate=training_params.decay_rate,
        period_epochs=training_params.period_epochs,
        
        # Optimization & regularization
        optimizer=training_params.optimizer,
        batch_size=training_params.batch_size,
        gradient_accumulation_steps=training_params.gradient_accumulation_steps,
        weight_decay=training_params.weight_decay,
        clip_grad_norm=training_params.clip_grad_norm,
        ema_decay=training_params.ema_decay,
        
        # Infrastructure
        seed=training_params.seed,
        run_dir=training_params.run_dir,
        overwrite_output_dir=training_params.overwrite_output_dir,
        
        # Evaluation
        primary_metric=training_params.primary_metric,
        
        # Early stopping
        patience=training_params.patience,
        min_delta=training_params.min_delta,
        early_stopping_sleep_epochs=training_params.early_stopping_sleep_epochs,

        # Training resumption
        resume_from_checkpoint=training_params.resume_from_checkpoint,
        reset_optimizer=training_params.reset_optimizer,
        
        # Data loading
        num_workers=training_params.num_workers,
        pin_memory=training_params.pin_memory,
        drop_last=training_params.drop_last,
        shuffle=training_params.shuffle,
        
        # Advanced training
        run_val_at_start=training_params.run_val_at_start,

        # debugging
        track_memory=training_params.track_memory,

        track_gradients=training_params.track_gradients,
        gradient_track_patterns=training_params.gradient_track_patterns,
        gradient_track_flow=training_params.gradient_track_flow,
    )
    
    return training_config


def load_and_split_dataset(params: ParamsObj) -> tuple[Subset, Subset]:
    """Load dataset and create train/validation splits."""
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
    """Main training function."""
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/2.3_training.log')
    
    logger.info("Starting metal site neural network training")
    
    # ============================================================================
    # STEP 1: Load DVC parameters for 2_pretraining
    # ============================================================================
    logger.info("Loading DVC parameters...")
    params = load_dvc_params()
    logger.info(f"Loaded parameters for 2_pretraining stage")
    
    # ============================================================================
    # STEP 2: Initialize model configuration, collator, and training config
    # ============================================================================
    logger.info("Initializing collator...")
    collator = initialize_collator(params)
    logger.info(f"Collator initialized with vocab sizes: {collator.featurizer.get_feature_vocab_sizes()}")
    
    logger.info("Initializing model configuration...")
    model_config = initialize_model_config(params, collator)
    logger.info(f"Model config initialized with atom_features={model_config.atom_features}, bond_features={model_config.bond_features}")
    
    logger.info("Initializing training configuration...")
    training_config = initialize_training_config(params)
    logger.info(f"Training config initialized: {training_config.max_epochs} epochs, batch_size={training_config.batch_size}")
    
    # ============================================================================
    # STEP 3: Load dataset and create train/validation splits
    # ============================================================================
    logger.info("Loading and splitting dataset...")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(params)
    logger.info(f"Dataset loaded and split successfully")
    
    # ============================================================================
    # STEP 4: Initialize trainer and run training
    # ============================================================================
    logger.info("Initializing MetalSiteTrainer...")
    trainer = MetalSiteTrainer(
        model_config=model_config,
        model_class=EquiformerWEdgesForPretraining,
        training_config=training_config,
        collator=collator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        custom_eval_fn=custom_eval_batch,
        custom_eval_log_fn=custom_eval_logger,
    )
    
    logger.info("Starting training...")
    trainer.run()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()