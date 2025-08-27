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
import os
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging
from pathlib import Path
from accelerate import Accelerator

from tqdm import tqdm

from metalsitenn.nn.pretrained_config import EquiformerWEdgesConfig
from metalsitenn.nn.model import ModelOutput, EquiformerWEdgesModel
from metalsitenn.featurizer import MetalSiteCollator
from metalsitenn.graph_data import BatchProteinData

import logging
logger = logging.getLogger(__name__)


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
    node_level_metrics: bool = False
    
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
    overwrite_output_dir: bool = False
    
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
        model_class: nn.Module,
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
        # attribute assignments
        self.model_config = model_config
        self.model_class = model_class
        self.training_config = training_config
        self.collator = collator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.eval_fn = eval_fn if eval_fn is not None else lambda batch, output: output

        # initialize training objects
        self._setup_output_dirs()
        self._setup_accelerator()
        self._setup_model()
        self._setup_data_loaders()
        self._setup_optimizer_and_scheduler()
        self._setup_logging_and_checkpointing()
        self._load_checkpoint_if_resuming()
    
    def _setup_output_dirs(self) -> None:
        """
        Create output directory, making sure to check for existence first.
        """
        if os.path.exists(self.training_config.run_dir):
            if self.training_config.overwrite_output_dir:
                logger.warning(f"Output directory {self.training_config.run_dir} exists and will be overwritten.")
                shutil.rmtree(self.training_config.run_dir)
            else:
                raise FileExistsError(f"Output directory {self.training_config.run_dir} already exists. "
                                      "Set overwrite_output_dir=True to overwrite.")
        os.makedirs(self.training_config.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training_config.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.training_config.run_dir, "dvclive"), exist_ok=True)
        logger.info(f"Created output directory at {self.training_config.run_dir}")

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
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            log_with='dvclive')
        self.accelerator.init_trackers(
            project_name="metal_site_trainer",
            init_kwargs={"dvclive": {
                "dir": self.training_config.run_dir + "/dvclive",
                "report": 'md',
                "save_dvc_exp": False,
                "dvcyaml": None
            }})


    def _setup_model(self) -> None:
        """
        Initialize and prepare the neural network model.
        
        Creates an EquiformerWEdgesForPretraining model using the provided model_config,
        initializes exponential moving average wrapper if ema_decay is specified in
        training_config, and prepares the model with accelerator.prepare() which handles
        device placement and distributed training wrapping automatically. Note that the EMA from fairchem
        looks like it is expected to be given POST parallelized model, so call accelerate first
        """
        assert issubclass(self.model_class, EquiformerWEdgesModel), "model_class must be EquiformerWEdgesModel or subclass"
        self.model = None
        self.ema_object = None
        logger.warning("`_setup_model` dry run called.")
    
    def _setup_data_loaders(self) -> None:
        """
        Create and prepare PyTorch DataLoaders.
        
        Constructs DataLoader objects for training, validation, and test datasets using
        the provided collator, configures batch size, workers, and sampling settings from
        training_config, and prepares data loaders with accelerator.prepare() which sets up
        distributed samplers and handles data loading optimizations automatically.
        """
        # Create training data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=self.training_config.shuffle,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            drop_last=self.training_config.drop_last,
            collate_fn=self.collator
        )
        
        # Create validation data loader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,  # Never shuffle validation
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            drop_last=False,  # Keep all validation data
            collate_fn=self.collator
        )
        
        # Create test data loader if test dataset provided
        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                drop_last=False,
                collate_fn=self.collator
            )
        else:
            self.test_loader = None
        
        logger.info(f"Created data loaders: train={len(self.train_loader)} batches, "
                   f"val={len(self.val_loader)} batches" + 
                   (f", test={len(self.test_loader)} batches" if self.test_loader else ""))
        
        # Note: accelerator.prepare() will be called later in setup to handle distributed sampling
        
    
    def _setup_optimizer_and_scheduler(self) -> None:
        """
        Initialize and prepare optimizer and learning rate scheduler.
        
        Creates optimizer (typically AdamW) using training_config parameters, initializes
        learning rate scheduler based on training_config.scheduler settings, configures
        warmup schedule if specified, and prepares both with accelerator.prepare() which
        handles optimizer state distribution and synchronization in distributed settings.
        """
        self.optimizer = None
        self.scheduler = None
        logger.warning("`_setup_optimizer_and_scheduler` dry run called.")
    
    def _setup_logging_and_checkpointing(self) -> None:
        """
        Configure logging and checkpoint management.
        
        Sets up logging infrastructure that respects accelerator's process rank (only main
        process logs to avoid duplication), creates checkpoint and log directories, initializes
        checkpoint tracking for best models based on primary_metric, and configures early
        stopping mechanism. Uses accelerator utilities for distributed-aware logging.

        Most of the logging setup MAY BE HANDLED BY DVC CALLBACK IN ACCELERATE. Not sure.
        """
        self.global_step = 0
        self.best_metric = None
        self.best_step = None
        self.evals_no_improve = 0
        self._metrics_cache = {}
        logger.warning("`_setup_logging_and_checkpointing` dry run called.")
    
    def _load_checkpoint_if_resuming(self) -> None:
        """
        Load checkpoint using Accelerate utilities.
        
        Uses accelerator.load_state() to restore model, optimizer, scheduler, and RNG states
        from checkpoint if resume_from_checkpoint is specified, handles distributed loading
        automatically, optionally resets optimizer state if reset_optimizer is True, and
        restores training metadata for continuing from correct epoch/step.
        """
        logger.warning("`_load_checkpoint_if_resuming` dry run called.")
    
    def run(self) -> None:
        """
        Execute the main training loop with Accelerate management.
        
        Runs complete training process with accelerator handling all infrastructure concerns,
        uses accelerator.backward() for gradient computation, accelerator.clip_grad_norm_() for
        gradient clipping, and accelerator utilities for metric gathering and synchronization.
        Handles distributed training, mixed precision, and gradient accumulation transparently
        through accelerator methods.
        """
        # TODO
        self._train_epoch(1)
    
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
        # self.model.train()
        
        # Initialize epoch metrics
        epoch_metrics = {}
        total_loss = 0.0
        num_batches = 0
        
        
        # Progress bar (only on main process)
        if self.accelerator.is_main_process:
            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
                disable=False
            )
        else:
            pbar = enumerate(self.train_loader)
        
        for batch_idx, batch in pbar:

            # take step
            _ = self._train_step(batch)


            #check for eval, checkpoin, early stop, etc
            # TODO

        return None

    def _track_metrics(self, metrics: Dict[str, torch.Tensor], final: bool=False) -> None:
        """
        Helps average training metrics since we will not log every step

        Args:
            metrics: Dictionary containing training 
            final: Whether this is the final call and we want to get averages and reset the tracker
        """
        for key, value in metrics.items():
            if key not in self._metrics_cache:
                self._metrics_cache[key] = []
            # Keep as tensors until final averaging
            self._metrics_cache[key].append(value)  # No .tolist()

        if final:
            avg_metrics = {}
            for key in self._metrics_cache.keys():
                if self.training_config.node_level_metrics and key != 'num_nodes':
                    # Concatenate all batches, weight by num_nodes
                    all_values = torch.cat(self._metrics_cache[key])
                    all_nodes = torch.cat(self._metrics_cache['num_nodes'])
                    weighted_sum = (all_values * all_nodes).sum()
                    total_nodes = all_nodes.sum()
                    avg_metrics[key] = (weighted_sum / total_nodes).item()
                else:
                    all_values = torch.cat(self._metrics_cache[key])
                    avg_metrics[key] = all_values.mean().item()

            self._metrics_cache = {}
            return avg_metrics
        return None
        
    
    def _train_step(
        self, 
        batch: BatchProteinData, 
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute one training step using Accelerate utilities.
        
        Args:
            batch: Batch of molecular data (already moved to correct device by accelerator)
            
        Performs forward pass through model with automatic device placement and mixed precision,
        extracts loss from ModelOutput, uses accelerator.backward() for gradient computation
        which handles scaling and distributed synchronization, updates EMA if enabled.
        """
        n_nodes = len(batch.element)

        # take a step in this class
        self.global_step += 1
        # take dvc live step
        # TODO

        with self.accelerator.accumulate(self.model):
            outs = ModelOutput(
                loss=torch.tensor(0.0)
            )

            # update training related classes
            # check outs for correct format
            assert outs.loss is not None, "ModelOutput must contain loss for training step"
            loss = outs.loss.item()
            # self.accelerator.backward(outs.loss)
            # self.optimizer.step()
            # self.scheduler.step()
            # self.optimizer.zero_grad()

        # TODO EMA update if needed


        # log train metrics
        # not not a logging step, add them to a cache
        # if logging step average since last log and actually do the logging
        # gather metrics across processes using accelerator.gather_for_metrics()
        # if node level values attempt to do a weighted average by num nodes
        train_metrics = {}
        for name, value in asdict(outs).items():
            if (name.endswith("loss") or name == "loss") and value is not None:
                value = value.detach().cpu().unsqueeze(0)  # Always make [1] tensor
                train_metrics[f"{name}"] = self.accelerator.gather_for_metrics(value)

        train_metrics['num_nodes'] = self.accelerator.gather_for_metrics(
            torch.tensor([n_nodes], dtype=torch.float)
        )

        if self.global_step % self.training_config.log_every == 0:
            agg_train_metrics = self._track_metrics(train_metrics, final=True)
            self._log_metrics(agg_train_metrics, prefix="train/")
        else:
            _ = self._track_metrics(train_metrics, final=False)

        return None

    
    def _validate(self) -> Dict[str, float]:
        """
        Run validation using Accelerate for distributed coordination.
        
        Returns:
            Dictionary containing validation metrics
            
        Sets model to evaluation mode, iterates through validation DataLoader with automatic
        device handling, performs forward passes with accelerator's automatic mixed precision,
        applies custom eval_fn if provided, uses accelerator.gather() to collect results from
        all processes in distributed setting, and returns aggregated validation metrics..
        """
        logger.warning("`_validate` dry run called.")
        return {'loss': 0.0}
    
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
        raise NotImplementedError("Batch evaluation not implemented yet.")
    
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
        raise NotImplementedError("Metric aggregation not implemented yet.")
    
    def _should_checkpoint(
        self, 
        val_metrics: Dict[str, float], 
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
        raise NotImplementedError("Checkpoint decision not implemented yet.")
    
    def _save_checkpoint(
        self, 
        val_metrics: Dict[str, float], 
        is_best: bool = False
    ) -> None:
        """
        Save checkpoint using Accelerate state management.
        
        Args:
            val_metrics: Current validation metrics
            is_best: Whether this is the best model so far
            
        Uses accelerator.save_state() to save model, optimizer, scheduler, and RNG states
        with proper handling of distributed training (only main process saves), manages
        checkpoint rotation according to max_checkpoints, saves additional best checkpoint
        if is_best is True, and includes metadata for resuming training.
        """
        raise NotImplementedError("Checkpoint saving not implemented yet.")
    
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
        raise NotImplementedError("Early stopping decision not implemented yet.")
    
    def _log_metrics(
        self, 
        metrics: Dict[str, float], 
        prefix: str = ""
    ) -> None:
        """
        Log metrics using Accelerate's distributed-aware logging.
        
        Args:
            metrics: Dictionary of metrics to logr
            prefix: Prefix for metric names (e.g., "train/", "val/")
            DVC callback should place them in appropriate folders.
            
        Uses accelerator.is_main_process to ensure only main process logs to avoid duplication,
        formats metrics for readable output, writes to configured logging handlers, includes
        epoch information, and integrates with accelerator's logging utilities for
        consistent distributed behavior.
        """
        logger.warning(f"`_log_metrics` dry run called on metrics: {metrics} with prefix: {prefix}")    
    
    def _cleanup(self) -> None:
        """
        Cleanup using Accelerate utilities.
        
        Uses accelerator.end_training() to properly clean up distributed processes and
        resources, closes logging handlers, performs final checkpoint save if needed,
        and handles graceful shutdown of all infrastructure managed by accelerator.
        """
        logger.warning("`_cleanup` dry run called.")