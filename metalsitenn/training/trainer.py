# metalsitenn/training/trainer.py
'''
* Author: Evan Komp
* Created: 8/21/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
"""
Trainer configuration dataclass for flexible neural network training.

This module provides a comprehensive configuration class for training neural networks,
particularly for molecular/protein modeling tasks. The configuration is designed to be
task-agnostic while providing all necessary hyperparameters for training control.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch


@dataclass
class TrainerConfig:
    """
    Configuration class for neural network training.
    
    This dataclass contains all hyperparameters needed for training neural networks
    in a flexible, task-agnostic manner. Task-specific parameters (like loss functions)
    should be handled by trainer subclasses.
    
    Attributes:
        # Core Training Parameters
        max_epochs (int): Maximum number of training epochs.
        eval_every (Optional[int]): Frequency of evaluation/checkpointing in steps. 
            Defaults to len(train_loader) if None.
        log_every (int): Frequency of logging in steps.
        max_checkpoints (int): Maximum number of best checkpoints to keep.
        save_best_only (bool): If True, only save checkpoints when validation improves.
        
        # Learning Rate & Scheduling
        lr_initial (float): Initial learning rate.
        scheduler (str): Learning rate scheduler type.
        lambda_type (str): Type of lambda function for LambdaLR scheduler.
        warmup_epochs (int): Number of epochs for learning rate warmup.
        warmup_factor (float): Starting factor for warmup (lr_initial * warmup_factor).
        lr_min_factor (float): Minimum LR factor for cosine annealing.
        decay_epochs (Optional[List[int]]): Epochs for multistep LR decay.
        decay_rate (float): Decay factor for multistep scheduler.
        
        # Optimization & Regularization
        optimizer (str): Optimizer class name.
        batch_size (int): Training batch size.
        gradient_accumulation_steps (int): Steps for gradient accumulation.
        weight_decay (float): L2 regularization weight decay.
        clip_grad_norm (Optional[float]): Gradient clipping threshold.
        ema_decay (Optional[float]): Exponential moving average decay rate.
        
        # Infrastructure
        amp (bool): Enable automatic mixed precision training.
        device (str): Device for training ('auto', 'cuda', 'cpu').
        seed (Optional[int]): Random seed for reproducibility.
        is_debug (bool): Enable debug mode.
        run_dir (Optional[str]): Directory for saving runs.
        checkpoint_dir (Optional[str]): Directory for saving checkpoints.
        
        # Evaluation
        primary_metric (str): Primary metric for model selection.
        
        # Early Stopping
        patience (Optional[int]): Early stopping patience (evaluations without improvement).
        min_delta (float): Minimum change to qualify as improvement.
        
        # Training Resumption
        resume_from_checkpoint (Optional[str]): Path to checkpoint for resuming training.
        reset_optimizer (bool): Reset optimizer state when resuming from checkpoint.
        
        # Data Loading & Sampling
        num_workers (int): Number of dataloader worker processes.
        pin_memory (bool): Pin memory for faster GPU transfer.
        drop_last (bool): Drop incomplete batches during training.
        shuffle (bool): Shuffle training data.
        
        # Advanced Training
        run_val_at_start (bool): Run validation before starting training.
    """
    
    # Core Training Parameters
    max_epochs: int = 100
    eval_every: Optional[int] = None
    log_every: int = 100
    max_checkpoints: int = 3
    save_best_only: bool = False
    
    # Learning Rate & Scheduling
    lr_initial: float = 1e-4
    scheduler: str = "LambdaLR"
    lambda_type: str = "cosine"
    warmup_epochs: int = 0
    warmup_factor: float = 0.2
    lr_min_factor: float = 0.01
    decay_epochs: Optional[List[int]] = None
    decay_rate: float = 0.1
    
    # Optimization & Regularization  
    optimizer: str = "AdamW"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None
    ema_decay: Optional[float] = None
    
    # Infrastructure
    amp: bool = False
    seed: Optional[int] = None
    is_debug: bool = False
    run_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    ddp: bool = False
    
    # Evaluation
    primary_metric: str = "val_loss"
    
    # Early Stopping
    patience: Optional[int] = None
    min_delta: float = 0.0
    
    # Training Resumption
    resume_from_checkpoint: Optional[str] = None
    reset_optimizer: bool = False
    
    # Data Loading & Sampling
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    
    # Advanced Training
    run_val_at_start: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        
        if self.lr_initial <= 0:
            raise ValueError("lr_initial must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be positive")
        
        if self.max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")
        
        if self.scheduler == "LambdaLR" and self.lambda_type not in ["cosine", "multistep"]:
            raise ValueError("lambda_type must be 'cosine' or 'multistep' for LambdaLR")
        
        if self.lambda_type == "multistep" and self.decay_epochs is None:
            raise ValueError("decay_epochs required for multistep scheduler")

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrainerConfig":
        """Create config from dictionary."""
        # Filter out unknown keys
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)