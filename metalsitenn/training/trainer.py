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
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging
from pathlib import Path
from accelerate import Accelerator

from metalsitenn.nn.pretrained_config import EquiformerWEdgesConfig
from metalsitenn.nn.model import ModelOutput
from metalsitenn.featurizer import MetalSiteCollator
from metalsitenn.graph_data import BatchProteinData


@dataclass
class TrainerConfig:
    """
    Configuration class for neural network training hyperparameters.
    
    Infrastructure (device, distributed, mixed precision) is handled by Accelerate config.
    This class focuses purely on training algorithm parameters.
    """
    
    # === CORE TRAINING ===
    max_epochs: int = 100
    eval_every: Optional[int] = None  # Steps between evaluations (defaults to epoch end)
    log_every: int = 10  # Steps between logging
    max_checkpoints: int = 5  # Maximum number of checkpoints to keep
    save_best_only: bool = True  # Only save when validation improves
    
    # === LEARNING RATE & SCHEDULING ===
    lr_initial: float = 1e-4
    scheduler: str = "LambdaLR"  # "LambdaLR", "CosineAnnealingLR", "StepLR"
    lambda_type: str = "cosine"  # "cosine", "multistep" for LambdaLR
    warmup_epochs: int = 0
    warmup_factor: float = 0.2  # Starting factor for warmup
    lr_min_factor: float = 0.01  # Minimum LR factor for cosine annealing
    decay_epochs: Optional[List[int]] = None  # Epochs for multistep decay
    decay_rate: float = 0.1  # Decay factor for multistep
    
    # === OPTIMIZATION & REGULARIZATION ===
    optimizer: str = "AdamW"
    batch_size: int = 32
    gradient_accumulation_steps: Optional[int] = None  # None = no accumulation
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None  # None = no clipping
    ema_decay: Optional[float] = None  # None = no EMA
    
    # === EVALUATION ===
    primary_metric: str = "val_loss"  # Metric for model selection
    
    # === EARLY STOPPING ===
    patience: Optional[int] = None  # None = no early stopping
    min_delta: float = 0.0  # Minimum improvement threshold
    
    # === TRAINING RESUMPTION ===
    resume_from_checkpoint: Optional[str] = None
    reset_optimizer: bool = False  # Reset optimizer state when resuming
    
    # === DATA LOADING ===
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    
    # === ADVANCED TRAINING ===
    run_val_at_start: bool = False
    seed: Optional[int] = None
    
    # === PATHS ===
    run_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        
        if self.lr_initial <= 0:
            raise ValueError("lr_initial must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
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


class MetalSiteTrainer:
    """
    Trainer for metal site neural networks using Accelerate for infrastructure management.
    """
    
    def __init__(
        self,
        model_config: EquiformerWEdgesConfig,
        training_config: TrainerConfig,
        collator: MetalSiteCollator,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        eval_fn: Optional[Callable[[BatchProteinData, ModelOutput], ModelOutput]] = None
    ) -> None:
        """
        Initialize the MetalSiteTrainer with Accelerate infrastructure.
        
        Args:
            model_config: Configuration for the EquiformerWEdges model
            training_config: Configuration for training hyperparameters
            collator: Data collator for batching molecular data
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset for final evaluation
            eval_fn: Optional custom evaluation function that takes (batch, model_output) -> ModelOutput.
                    If None, uses default pass-through evaluation that returns model_output unchanged.
        
        Stores all configuration objects and datasets, initializes Accelerator which handles
        device placement, distributed training, and mixed precision automatically based on
        accelerate config. Creates the model from model_config, sets up data loaders using
        the provided collator, initializes optimizer and scheduler, sets up logging and
        checkpointing, prepares all components with accelerator.prepare(), and loads from
        checkpoint if resuming training is specified.
        """
        pass
    
    def _setup_accelerator(self) -> None:
        """
        Initialize Accelerate infrastructure.
        
        Creates Accelerator instance which reads configuration from accelerate config files
        or environment variables to determine device placement (CPU/GPU/TPU), distributed
        training setup (DDP, FSDP, DeepSpeed), and mixed precision settings. The accelerator
        handles all infrastructure concerns automatically, including process group initialization,
        device placement, gradient synchronization, and mixed precision scaling. Note we need
        to configure the DVC callback to the accelerator here so that when we log metrics
        etc. it goes to dvc
        """
        pass
    
    def _setup_model(self) -> None:
        """
        Initialize and prepare the neural network model.
        
        Creates an EquiformerWEdgesForPretraining model using the provided model_config,
        initializes exponential moving average wrapper if ema_decay is specified in
        training_config, and prepares the model with accelerator.prepare() which handles
        device placement and distributed training wrapping automatically. Note that the EMA from fairchem
        looks like it is expected to be given POST parallelized model, so call accelerate first
        """
        pass
    
    def _setup_data_loaders(self) -> None:
        """
        Create and prepare PyTorch DataLoaders.
        
        Constructs DataLoader objects for training, validation, and test datasets using
        the provided collator, configures batch size, workers, and sampling settings from
        training_config, and prepares data loaders with accelerator.prepare() which sets up
        distributed samplers and handles data loading optimizations automatically.
        """
        pass
    
    def _setup_optimizer_and_scheduler(self) -> None:
        """
        Initialize and prepare optimizer and learning rate scheduler.
        
        Creates optimizer (typically AdamW) using training_config parameters, initializes
        learning rate scheduler based on training_config.scheduler settings, configures
        warmup schedule if specified, and prepares both with accelerator.prepare() which
        handles optimizer state distribution and synchronization in distributed settings.
        """
        pass
    
    def _setup_logging_and_checkpointing(self) -> None:
        """
        Configure logging and checkpoint management.
        
        Sets up logging infrastructure that respects accelerator's process rank (only main
        process logs to avoid duplication), creates checkpoint and log directories, initializes
        checkpoint tracking for best models based on primary_metric, and configures early
        stopping mechanism. Uses accelerator utilities for distributed-aware logging.

        Most of the logging setup MAY BE HANDLED BY DVC CALLBACK IN ACCELERATE. Not sure.
        """
        pass
    
    def _load_checkpoint_if_resuming(self) -> None:
        """
        Load checkpoint using Accelerate utilities.
        
        Uses accelerator.load_state() to restore model, optimizer, scheduler, and RNG states
        from checkpoint if resume_from_checkpoint is specified, handles distributed loading
        automatically, optionally resets optimizer state if reset_optimizer is True, and
        restores training metadata for continuing from correct epoch/step.
        """
        pass
    
    def run(self) -> None:
        """
        Execute the main training loop with Accelerate management.
        
        Runs complete training process with accelerator handling all infrastructure concerns,
        uses accelerator.backward() for gradient computation, accelerator.clip_grad_norm_() for
        gradient clipping, and accelerator utilities for metric gathering and synchronization.
        Handles distributed training, mixed precision, and gradient accumulation transparently
        through accelerator methods.
        """
        pass
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch using Accelerate.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics for this epoch
            
        Iterates through training DataLoader prepared by accelerator, performs forward passes
        with automatic mixed precision if configured, uses accelerator.backward() for gradient
        computation which handles scaling and synchronization, accumulates gradients according
        to gradient_accumulation_steps, performs optimizer steps, updates scheduler, and logs
        metrics using accelerator's distributed-aware utilities.
        """
        pass
    
    def _train_step(
        self, 
        batch: BatchProteinData, 
        step: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute one training step using Accelerate utilities.
        
        Args:
            batch: Batch of molecular data (already moved to correct device by accelerator)
            step: Current step number
            
        Returns:
            Tuple of (loss_tensor, metrics_dict)
            
        Performs forward pass through model with automatic device placement and mixed precision,
        extracts loss from ModelOutput, uses accelerator.backward() for gradient computation
        which handles scaling and distributed synchronization, updates EMA if enabled, and
        returns loss and metrics with proper device handling.
        """
        pass
    
    def _validate(self, epoch: int, step: int) -> Dict[str, float]:
        """
        Run validation using Accelerate for distributed coordination.
        
        Args:
            epoch: Current epoch number
            step: Current step number
            
        Returns:
            Dictionary containing validation metrics
            
        Sets model to evaluation mode, iterates through validation DataLoader with automatic
        device handling, performs forward passes with accelerator's automatic mixed precision,
        applies custom eval_fn if provided, uses accelerator.gather() to collect results from
        all processes in distributed setting, and returns aggregated validation metrics.
        """
        pass
    
    def _evaluate_batch(
        self, 
        batch: BatchProteinData
    ) -> ModelOutput:
        """
        Evaluate single batch with custom evaluation function.
        
        Args:
            batch: Batch of molecular data (device placement handled by accelerator)
            
        Returns:
            ModelOutput containing evaluation results
            
        Performs forward pass through model with batch already on correct device via accelerator,
        applies custom eval_fn if provided during initialization (otherwise returns model output
        unchanged), handles any additional metric computation specified by eval_fn, and returns
        ModelOutput with computed metrics ready for aggregation across processes.
        """
        pass
    
    def _aggregate_metrics(
        self, 
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across processes using Accelerate.
        
        Args:
            metrics: Dictionary of metrics from current process
            
        Returns:
            Dictionary of aggregated metrics across all processes
            
        Uses accelerator.gather() to collect metric tensors from all processes, performs
        appropriate aggregation (mean, sum) based on metric type, handles synchronization
        automatically through accelerator utilities, and returns final metrics that represent
        performance across entire distributed system. Only main process receives final results.
        """
        pass
    
    def _should_checkpoint(
        self, 
        val_metrics: Dict[str, float], 
        epoch: int
    ) -> bool:
        """
        Determine checkpointing using distributed-aware comparison.
        
        Args:
            val_metrics: Validation metrics (already aggregated across processes)
            epoch: Current epoch number
            
        Returns:
            Boolean indicating whether to save checkpoint
            
        Compares current validation metrics against best seen so far using primary_metric,
        applies save_best_only logic and min_delta threshold, ensures consistent decisions
        across all processes in distributed setting, and returns checkpoint decision that
        is synchronized across all ranks.
        """
        pass
    
    def _save_checkpoint(
        self, 
        val_metrics: Dict[str, float], 
        epoch: int, 
        step: int,
        is_best: bool = False
    ) -> None:
        """
        Save checkpoint using Accelerate state management.
        
        Args:
            val_metrics: Current validation metrics
            epoch: Current epoch number
            step: Current step number
            is_best: Whether this is the best model so far
            
        Uses accelerator.save_state() to save model, optimizer, scheduler, and RNG states
        with proper handling of distributed training (only main process saves), manages
        checkpoint rotation according to max_checkpoints, saves additional best checkpoint
        if is_best is True, and includes metadata for resuming training.
        """
        pass
    
    def _should_early_stop(
        self, 
        val_metrics: Dict[str, float]
    ) -> bool:
        """
        Check early stopping with distributed coordination.
        
        Args:
            val_metrics: Current validation metrics (aggregated across processes)
            
        Returns:
            Boolean indicating whether to stop training early
            
        Tracks validation improvements over recent epochs using primary_metric, compares
        against patience threshold and min_delta, maintains counter of epochs without
        improvement, ensures early stopping decision is consistent across all processes
        in distributed setting, and returns synchronized early stopping signal.
        """
        pass
    
    def _log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: int, 
        prefix: str = ""
    ) -> None:
        """
        Log metrics using Accelerate's distributed-aware logging.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step number
            prefix: Prefix for metric names (e.g., "train/", "val/")
            DVC callback should place them in appropriate folders.
            
        Uses accelerator.is_main_process to ensure only main process logs to avoid duplication,
        formats metrics for readable output, writes to configured logging handlers, includes
        step/epoch information, and integrates with accelerator's logging utilities for
        consistent distributed behavior.
        """
        pass
    
    def _cleanup(self) -> None:
        """
        Cleanup using Accelerate utilities.
        
        Uses accelerator.end_training() to properly clean up distributed processes and
        resources, closes logging handlers, performs final checkpoint save if needed,
        and handles graceful shutdown of all infrastructure managed by accelerator.
        """
        pass