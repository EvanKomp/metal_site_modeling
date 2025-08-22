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

logger = logging.getLogger(__name__)

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
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
                model_params['node_class_weights'] = weights_data['class_weights']
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
        checkpoint_dir=training_params.checkpoint_dir,
        
        # Evaluation
        primary_metric=training_params.primary_metric,
        
        # Early stopping
        patience=training_params.patience,
        min_delta=training_params.min_delta,
        
        # Training resumption
        resume_from_checkpoint=training_params.resume_from_checkpoint,
        reset_optimizer=training_params.reset_optimizer,
        
        # Data loading
        num_workers=training_params.num_workers,
        pin_memory=training_params.pin_memory,
        drop_last=training_params.drop_last,
        shuffle=training_params.shuffle,
        
        # Advanced training
        run_val_at_start=training_params.run_val_at_start
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
        training_config=training_config,
        collator=collator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    logger.info("Starting training...")
    trainer.run()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()