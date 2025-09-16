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
import functools
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
from accelerate import Accelerator

from tqdm import tqdm

from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage

from metalsitenn.nn.pretrained_config import EquiformerWEdgesConfig
from metalsitenn.nn.model import ModelOutput, EquiformerWEdgesModel
from metalsitenn.featurizer import MetalSiteCollator
from metalsitenn.graph_data import BatchProteinData
from metalsitenn.training.mem_tracking import TorchTracemalloc
from metalsitenn.training.gradient_tracker import GradientTracker
from metalsitenn.training.lr_scheduler_factory import create_scheduler

from metalsitenn.nn.debug_module_pretraining import SimpleDebugModel

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
    period_epochs: float = 0.5
    
    # === OPTIMIZATION & REGULARIZATION ===
    optimizer: str = "AdamW"
    batch_size: int = 32
    gradient_accumulation_steps: Optional[int] = None  # None = no accumulation
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None  # None = no clipping
    ema_decay: Optional[float] = None  # None = no EMA
    
    # === EVALUATION ===
    primary_metric: str = "val_loss"  # Metric for model selection
    primary_metric_mode: str = "min"  # one of 'min', 'max'
    node_level_metrics: bool = False
    
    # === EARLY STOPPING ===
    patience: Optional[int] = None  # None = no early stopping
    min_delta: float = 0.0  # largest (most positive) change in tracked eval metric to not be considered a failed change
    early_stopping_sleep_epochs: float = 0.2  # Number of epochs to wait before starting to consider an early stopping

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

    # === DEBUGGING ===
    track_memory: bool = False

    track_gradients: bool = True
    gradient_track_patterns: Optional[List[str]] = None  # None = all layers, otherwise list of layer names to track
    gradient_track_flow: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        
        if self.lr_initial <= 0:
            raise ValueError("lr_initial must be positive")
        
        if self.early_stopping_sleep_epochs < 0:
            raise ValueError("early_stopping_sleep_epochs must be non-negative")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if self.max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")
        
        if self.scheduler == "LambdaLR" and self.lambda_type not in ["cosine", "multistep", "multistep_cosine", "warm_restart_cosine"]:
            raise ValueError("lambda_type must be 'cosine' or 'multistep' or 'multistep_cosine' for LambdaLR")

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

######################################################################
# HELPER FOR AVOIDING DUPLICATE WORK ACROSS PROCESSES
######################################################################

def main_process_only(func: Callable) -> Callable:
    """Decorator that only executes the method on the main process."""
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            return func(self, *args, **kwargs)
        return None
    return wrapper

######################################################################
# Early stopping exception
######################################################################
class EarlyStoppingException(Exception):
    """Custom exception for early stopping."""
    pass

######################################################################
# Used of user does not provide to convert metric results from all batches in
# the eval set to a final output dict and log statements.
# Note, custom behavior must call trainer._log_metrics if it wants DVC to be tracking its eval quantities.
######################################################################

def _default_eval_log_fn(trainer, metrics: Dict[str, torch.Tensor]):
    """
    Default evaluation logging function that averages metrics.
    
    Args:
        trainer: The trainer instance
        metrics: Dictionary of metric_name -> tensor from evaluation
        
    Returns:
        None
    """
    agg_metrics = {}
    for key, list_of_tensors in metrics.items():
        # Concatenate all tensors along first dimension
        all_values = torch.cat(list_of_tensors)
        if len(all_values) != (len(trainer.val_loader)*trainer.accelerator.num_processes):
            if trainer.accelerator.is_main_process:
                logger.warning(f"Eval metric {key} does not have the length equal to the total number of eval batches, but is going to be meaned anyway. It may not have a signficant meaning, shape is {all_values.shape}")

        mean_value = all_values.float().mean().item()
        agg_metrics[key] = mean_value

    # do the logging
    trainer._log_metrics(agg_metrics, prefix="eval/")

    # also return metrics so that the upper processes have access to it
    return agg_metrics

######################################################################
# Main entry to training API
######################################################################

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
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        custom_eval_fn: Optional[Callable[[BatchProteinData, ModelOutput], Dict]] = None,
        custom_eval_log_fn: Optional[Callable[["MetalSiteTrainer", Dict[str, torch.Tensor]], Dict[str, float]]] = None,
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
            custom_eval_fn: Optional custom evaluation function that takes (batch, model_output) -> dict of quantities (possbily full tensors).
                    If None, uses default pass-through evaluation that returns loss from model output.
                    NOTE: These will have to be gathered across processes - if you return a per-batch quantity
                    It will be gathered to a size N_devices tensor. If you return tensors with more than one quantity
                    they will be padded and then gathered along the first dimension.
            custom_eval_log_fn: Optional function to log full eval evaluation metrics. By default its just going to cat and mean the quantities
                    and log them.
        
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
        self.custom_eval_fn = custom_eval_fn if custom_eval_fn is not None else lambda batch, output: getattr(output, 'loss', None)
        self.custom_eval_log_fn = custom_eval_log_fn if custom_eval_log_fn is not None else _default_eval_log_fn
        

        # initialize training objects
        self._setup_accelerator() # first so that we can get access to is_main_process
        self._base_logger = logger
        self._setup_output_dirs()
        self._setup_model()
        self._setup_data_loaders()
        self._setup_optimizer_and_scheduler()
        self._accelerate_prepare_model_optimizer_scheduler()
        self._setup_ema()
        self._setup_logging_and_checkpointing()
        self._setup_gradient_tracking()
        self._load_checkpoint_if_resuming()

        self.log_info(f"Training configuration: {self.training_config.to_dict()}")

    @main_process_only
    def log_info(self, msg):
        self._base_logger.info(msg)

    @main_process_only
    def log_warning(self, msg):
        self._base_logger.warning(msg)

    @main_process_only
    def log_debug(self, msg):
        self._base_logger.debug(msg)

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
            step_scheduler_with_optimizer=True,
            log_with='dvclive')
        self.accelerator.init_trackers(
            project_name="metal_site_trainer",
            init_kwargs={"dvclive": {
                "dir": self.training_config.run_dir + "/dvclive",
                "report": 'md',
                "save_dvc_exp": False,
                "dvcyaml": None
            }})
        
        self.accelerator.wait_for_everyone()

    @main_process_only
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
        # self.model = SimpleDebugModel(
        #     vocab_size=self.model_config.feature_vocab_sizes['element'],
        #     cel_class_weights=torch.Tensor(self.model_config.node_class_weights),
        #     label_smoothing=self.model_config.node_class_label_smoothing
        # )
        self.model = self.model_class(self.model_config)

        num_trainable_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                num_trainable_params += param.numel()
        self.log_info(f"Model initialized with {num_trainable_params} trainable parameters.")

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
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,  # Never shuffle validation
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                drop_last=False,  # Keep all validation data
                collate_fn=self.collator
            )
        else:
            self.val_loader = None
        
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


        self.log_info(f"Created data loaders: train={len(self.train_loader)} total batches, "
                f"val={len(self.val_loader)} total batches" + 
                (f", test={len(self.test_loader)} total batches" if self.test_loader else ""))
        
        # send the datasets to their respective devices
        self.train_loader = self.accelerator.prepare(self.train_loader)
        if self.val_dataset is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)
        if self.test_dataset is not None:
            self.test_loader = self.accelerator.prepare(self.test_loader)

        # log new length
        num_steps = len(self.train_loader)
        effective_step_size = self.training_config.batch_size * self.accelerator.num_processes
        self.log_info(f"On device training loader={num_steps} total steps of size {effective_step_size}")

        # also log gradient accumulation related stats, maybe this should go in a different method
        effective_batch_size = effective_step_size * (self.training_config.gradient_accumulation_steps or 1)
        self.log_info(f"Effective batch size per step (all devices + accumulation) = {effective_batch_size}")

        self.accelerator.wait_for_everyone()
        
    
    def _setup_optimizer_and_scheduler(self) -> None:
        """
        Initialize and prepare optimizer and learning rate scheduler.
        
        Creates optimizer (typically AdamW) using training_config parameters, initializes
        learning rate scheduler based on training_config.scheduler settings, configures
        warmup schedule if specified, and prepares both with accelerator.prepare() which
        handles optimizer state distribution and synchronization in distributed settings.
        """
        lr = self.training_config.lr_initial * self.accelerator.num_processes
        self.log_info(f"Setting initial learning rate to {lr} (scaled by number of processes)")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # now scheduler
        # huggingface will "automagically" (I really dislike this behavior)
        # do extra steps in the prepared scheduler based on num processes
        # this means we need to compute with pre-prepared dataset sizes
        # BUT the scheduler is only stepped with the optimizer, so even if it is doing Nprocess steps
        # it may be way behind on the schdule if accumulation is being used, 
        # so we need to reduce the total steps by gradient accumulation steps...
        steps_per_epoch_raw = len(self.train_loader) * self.accelerator.num_processes
        steps_per_epoch_with_accumulation = steps_per_epoch_raw // (self.training_config.gradient_accumulation_steps or 1)

        total_steps = steps_per_epoch_with_accumulation * self.training_config.max_epochs
        warmup_steps = int(steps_per_epoch_with_accumulation * self.training_config.warmup_epochs)
        period = int(steps_per_epoch_with_accumulation * self.training_config.period_epochs)

        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.training_config.scheduler,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_factor=self.training_config.warmup_factor,
            lr_min_factor=self.training_config.lr_min_factor,
            decay_epochs=self.training_config.decay_epochs,
            decay_rate=self.training_config.decay_rate,
            lambda_type=self.training_config.lambda_type,
            period=period
        )


    def _accelerate_prepare_model_optimizer_scheduler(self) -> None:
        """
        Prepare model, optimizer, and learning rate scheduler with accelerator.
        """
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _setup_ema(self) -> None:
        """
        Setup Exponential Moving Average (EMA) for model parameters.
        """
        if self.training_config.ema_decay:
            self.ema_object = ExponentialMovingAverage(self.model.parameters(), decay=self.training_config.ema_decay)
            self.accelerator.register_for_checkpointing(self.ema_object)
        else:
            self.ema_object = None

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


    def _setup_gradient_tracking(self) -> None:
        """Setup gradient tracking for debugging."""
        if self.training_config.track_gradients:
            self.gradient_tracker = GradientTracker(
                model=self.model,
                track_patterns=self.training_config.gradient_track_patterns,
                track_flow=self.training_config.gradient_track_flow,
            )
            self.log_info(f"Gradient tracking enabled for {len(self.gradient_tracker.get_tracked_groups())} layers")
        else:
            self.gradient_tracker = None

    def _load_checkpoint_if_resuming(self, checkpoint_path: str=None) -> None:
        """
        Load checkpoint using Accelerate utilities.
        
        Uses accelerator.load_state() to restore model, optimizer, scheduler, and RNG states
        from checkpoint if resume_from_checkpoint is specified, handles distributed loading
        automatically, optionally resets optimizer state if reset_optimizer is True, and
        restores training metadata for continuing from correct epoch/step.
        """
        if checkpoint_path is None:
            checkpoint_path = self.training_config.resume_from_checkpoint

        if checkpoint_path is None:
            return

        # try to parse the step number from the checkpoint
        # it should follow the form '...checkpoint_<step>'
        step_num = str(checkpoint_path).split("checkpoint_")[-1]
        step_num = int(step_num) if step_num.isdigit() else None
        if step_num is None:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}, cannot parse global step")

        # set global step to this point
        # should this be set to plus 1? I dont think so because we use it to update the dataloader to this
        # current point before first train step, then train step immediately ticks it
        self.global_step = step_num

        # actually load the checkpoint
        self.accelerator.load_state(checkpoint_path)

    def run(self) -> None:
        """
        Execute the main training loop with Accelerate management.
        
        Runs complete training process. Its assumed that any resumption has already taken place by
        _load_checkpoint_if_resuming() during init. Here we will jump the dataloader if we are resuming.
        Note that if the batch size or number of devices changes since the loaded checkpoint this data jumping
        is not rigorous anymore. Then begin training at that stage
        """
        # do an eval at start
        self._validate()

        if self.global_step != 0:
            # get steps per epoch
            steps_per_epoch = len(self.train_loader)
            # get current epoch
            current_epoch = (self.global_step // steps_per_epoch) + 1
            # steps taken already in this epoch
            steps_to_skip = self.global_step % steps_per_epoch
            self.log_info(f"Resuming training from epoch {current_epoch}, step {steps_to_skip}, steps remaining in epoch {steps_per_epoch - steps_to_skip}")

            # patch over the dataloader for this first initial epoch
            skipped_train_dataloader = self.accelerator.skip_first_batches(
                self.train_loader, num_batches=steps_to_skip
            )
            self._train_loader = self.train_loader
            self.train_loader = skipped_train_dataloader # self._train_epoch uses this attribute

            # continue training at this point
            self._train_epoch(current_epoch)

            # Now we return you to your scheduled broadcast
            self.train_loader = self._train_loader

            # we just finished that epoch so can tick that
            current_epoch += 1
        else:
            current_epoch = 1

        # main loop
        for epoch in range(current_epoch, self.training_config.max_epochs + 1):
            try:
                self._train_epoch(epoch)
            except EarlyStoppingException:
                self.log_info(f"Early stopping triggered at epoch {epoch}, step {self.global_step}, best metric {self.best_metric}")
                break

        # final cleanup
        self._finalize_training()

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
        self.model.train()
        
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
        
        # start memory tracker
        mem_tracker = TorchTracemalloc(do_tracking=self.training_config.track_memory, device=self.accelerator.device)
        mem_tracker.start()

        for _, batch in pbar:
            
            # take step
            _ = self._train_step(batch)

            #check for eval
            if self._should_eval():

                self.accelerator.wait_for_everyone()
                # we have gotten to the end of a trainer phase, track mem
                mem_stats = mem_tracker.stop()
                if mem_stats is None:
                    pass
                else:
                    # we need to get them from across processes
                    mem_stats = self.accelerator.gather_for_metrics(mem_stats)
                    # max them, I think this is probably the best aggregator for this type of tracking
                    mem_stats = {k: v.max().item() for k, v in mem_stats.items()}
                    self._log_metrics(mem_stats, prefix="train/memory/")
                    # validate will handle its own mem tracking

                self.log_debug("Beginning validation ...")

                # swap out ema for eval if present
                if self.ema_object is not None:
                    self.log_info("Swapping to EMA for evaluation")
                    self.ema_object.store() # stores the current instantaneous params
                    self.ema_object.copy_to() # and replaces with the EMA

                eval_metrics = self._validate()
                self.log_info(f"Eval metrics at step {self.global_step}: {eval_metrics}")

                # swap back in instananeous params from EMA
                if self.ema_object is not None:
                    self.log_info("Restoring model params post EMA evaluation")
                    self.ema_object.restore() # restores the instantaneous params back to the model

                # we do not want ema params for checkpointing during training,
                # the ema itself is checkpointed. The only time we want to save the model with
                # the ema params is maybe the final save of the unsharded PretrainedModel

                # do some checks for tracking best checkpoint and early stopping
                if self.training_config.primary_metric is not None:
                    assert self.training_config.primary_metric in eval_metrics, \
                        f"Primary metric {self.training_config.primary_metric} not found in eval metrics"

                    tracked_eval_metric = eval_metrics[self.training_config.primary_metric]
                    # check that it is a single quantity
                    try:
                        tracked_eval_metric = float(tracked_eval_metric)
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid value for primary metric {self.training_config.primary_metric}: {tracked_eval_metric}")

                    # tick early stopping
                    should_stop = self._tick_early_stopping(tracked_eval_metric)

                    # keep track of the best metric, checkpoint will use this.
                    tracked_metric_lower_better = (self.training_config.primary_metric_mode == 'min')
                    if tracked_metric_lower_better and (self.best_metric is None or (tracked_eval_metric < (self.best_metric - self.training_config.min_delta))):
                        is_best = True
                        self.best_metric = tracked_eval_metric
                    elif not tracked_metric_lower_better and (self.best_metric is None or (tracked_eval_metric > (self.best_metric + self.training_config.min_delta))):
                        is_best = True
                        self.best_metric = tracked_eval_metric
                    else:
                        is_best = False

                    # now do a checkpoint
                    self._save_checkpoint(is_best=is_best)

                    # early stop
                    if should_stop:
                        raise EarlyStoppingException()
                
                else:
                    # apparently we are too maverick for keeping track of the best or early stopping
                    self._save_checkpoint() 

                # restart the training tracker in this method
                mem_tracker.start()

                # take a breath after eval
                self.accelerator.wait_for_everyone()

            else:
                # no eval
                pass

        self.log_info(f"Completed epoch {epoch}/{self.training_config.max_epochs}")
        return None
    
    def _train_step(
        self, 
        batch: BatchProteinData, 
    ) -> None:
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

        with self.accelerator.accumulate(self.model):
            outs = self.model(batch)

            # update training related classes
            # check outs for correct format
            assert outs.loss is not None, "ModelOutput must contain loss for training step"
            loss = outs.loss.item()
            self.accelerator.backward(outs.loss)

            # also get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']

            # get gradient related metrics
            grad_metrics = {}
            if self.gradient_tracker is not None:
                grad_metrics = self.gradient_tracker.compute_metrics()

            # clip
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.training_config.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # update the ema params on a grad update
        # ego only when we actually update params
        if self.ema_object is not None:
            if self.accelerator.sync_gradients:
                self.ema_object.update()


        # log train metrics
        # if not a logging step, add them to a cache
        # if logging step average since last log and actually do the logging
        # gather metrics across processes using accelerator.gather_for_metrics()
        # if node level values attempt to do a weighted average by num nodes
        train_metrics = {}
        for name, value in outs.__dict__.items():
            if (name.endswith("loss") or name == "loss") and value is not None:
                value = value.detach().unsqueeze(0)  # Always make [1] tensor
                train_metrics[f"{name}"] = self.accelerator.gather_for_metrics(value)

        train_metrics['num_nodes'] = self.accelerator.gather_for_metrics(
            torch.tensor([n_nodes], dtype=torch.float, device=self.accelerator.device)
        )

        # gather gradient metrics
        for grad_metric_name, grad_metric_value in grad_metrics.items():
            train_metrics[grad_metric_name] = self.accelerator.gather_for_metrics(grad_metric_value)

        # add lr
        train_metrics['lr'] = self.accelerator.gather_for_metrics(
            torch.tensor([current_lr], dtype=torch.float, device=self.accelerator.device)
        )

        if self.global_step % self.training_config.log_every == 0:
            agg_train_metrics = self._track_per_step_metrics(train_metrics, final=True)
            # pop num_nodes since it is really not a useful quantity to log
            agg_train_metrics.pop('num_nodes', None)
            self._log_metrics(agg_train_metrics, prefix="train/")
        else:
            _ = self._track_per_step_metrics(train_metrics, final=False)

        # let's not get ahead of ourselves
        self.accelerator.wait_for_everyone()
        return None
    
    def _track_per_step_metrics(self, metrics: Dict[str, torch.Tensor], final: bool=False) -> None:
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
            # checking that gether creates the correct shape on multiple devices

            self.log_debug(f"Gathered train metrics at step {self.global_step}: {self._metrics_cache}") 

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
                    
                    # checking that gether creates the correct shape on multiple devices

                    self.log_debug(f"Averaging {key} over tensor shape {all_values.shape}")

                    avg_metrics[key] = all_values.mean().item()

            self._metrics_cache = {}
            return avg_metrics
        return None
    
    def _validate(self) -> Dict[str, float]:
        """
        Run validation using Accelerate for distributed coordination.
        
        Returns:
            Dictionary containing validation metrics
            
        Sets model to evaluation mode, iterates through validation DataLoader with automatic
        device handling, performs forward passes with accelerator's automatic mixed precision,
        applies custom custom_eval_fn if provided, uses accelerator.gather() to collect results from
        all processes in distributed setting, and returns aggregated validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        # start memory tracker
        mem_tracker = TorchTracemalloc(do_tracking=self.training_config.track_memory, device=self.accelerator.device)
        mem_tracker.start()

        # set model to eval mode
        self.model.eval()
        # loop over batches and call evaluate batch (which will use custom eval fn if provided)
        if self.accelerator.is_main_process:
            pbar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Validation",
                disable=False
            )
        else:
            pbar = enumerate(self.val_loader)

        all_metrics = {}
        for batch_idx, batch in pbar:
            # these metrics are just on this device
            # they might be a single scaler such as a loss
            # or they might be per node quantities
            batch_metrics = self._evaluate_batch(batch)

            # gather them across processes keep appending to a dict of list from the results
            # If originally they were scalars they will now be a [n_processes] tensor
            # If originally they were per node they will be [total_nodes_all_processes, ... eg padded TODO: NEED TO CONFIRM THIS]
            # they also get sent to cpu here so we don't fill up gpu memory storing full eval results
            gathered_metrics = self._gather_metrics(batch_metrics)
            for key, tensor in gathered_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(tensor)

        self.log_debug(f"Gathered all eval metrics: {all_metrics.keys()}")
        for key, tensors_list in all_metrics.items():
            one_tensor = tensors_list[0]
            self.log_debug(f"Eval metric '{key}' first tensor shape: {one_tensor.shape}")

        # dict of list of tensors is now the expected format for the eval log fn
        # by defualt it will just cat and mean them
        all_metrics = self.custom_eval_log_fn(self, all_metrics)
        
        # convert model back to train mode
        self.model.train()

        # log memory metrics
        mem_stats = mem_tracker.stop()
        if mem_stats is None:
            pass
        else:
            mem_stats = self.accelerator.gather_for_metrics(mem_stats)
            mem_stats = {k: v.max().item() for k, v in mem_stats.items()}
            self._log_metrics(mem_stats, prefix="eval/memory/")

        self.accelerator.wait_for_everyone()
        return all_metrics
    
    def _evaluate_batch(
        self, 
        batch: BatchProteinData
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate single batch with custom evaluation function.
        
        Args:
            batch: Batch of molecular data (device placement handled by accelerator)
            
        Returns:
            Dict containing per batch metrics to be gathered across processes.
            
        Performs forward pass through model with batch already on correct device via accelerator,
        applies custom custom_eval_fn if provided during initialization,
        shapes the resulting tensors for distributed gathering.
        """
        model_outs = self.model(batch)

        metrics = self.custom_eval_fn(batch, model_outs)
        return metrics
    
    def _gather_metrics(
        self, 
        metrics: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Gather metrics across processes, handling both scalar and per-node tensors, places them on cpu.
        For per-node tensors, automatically removes padding after gathering.
        
        Args:
            metrics: Dict of metric_name -> tensor from current process
                    Tensors can be:
                    - Scalar (shape [] or [1]): single value per batch 
                    - BROKEN Per-node (shape [n_nodes] or [n_nodes, features]): values per node
                    - BROKEN Other? Will be padded and concated along 0th dimension
            
        Returns:
            Dict of gathered tensors ready for aggregation
            - Scalar tensors: [num_processes] shape, ready for mean/sum
            - Per-node tensors: [total_valid_nodes_all_processes, ...] shape with padding removed
        """
        gathered_metrics = {}
        
        for key, tensor in metrics.items():
            if tensor is None:
                continue
                
            # Ensure tensor is actually a tensor
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, device=self.accelerator.device)
            
            # Detect if this is a scalar or per-node metric
            is_scalar = tensor.numel() == 1 or (tensor.ndim >= 1 and tensor.shape[0] == 1)
            
            if is_scalar:
                # Scalar metric - use gather_for_metrics
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                elif tensor.shape[0] != 1:
                    tensor = tensor.mean().unsqueeze(0)
                    
                gathered_metrics[key] = self.accelerator.gather_for_metrics(tensor).cpu()
                
            else:
                # self.log_warning(f"Non batch-level metrics (eg. more than one quantity per batch): ({key}) with shape ({tensor.shape}) are not currently must be the same shape due to `accelerator.pad_across_proceses` hanging.")

                gathered_metrics[key] = self.accelerator.gather_for_metrics(tensor).cpu()
                # self.log_debug(f"Gathered metric '{key}': shape {gathered_metrics[key].shape}")
                # # Per-node metric - pad, gather, then remove padding
                # original_length = tensor.shape[0]
                
                # # Create validity mask for this process (all True initially)
                # validity_mask = torch.ones(original_length, dtype=torch.bool, device=tensor.device)
                
                # try:
                #     # Pad tensor and validity mask
                #     padded_tensor = self.accelerator.pad_across_processes(tensor, dim=0)
                #     padded_mask = self.accelerator.pad_across_processes(validity_mask, dim=0)
                    
                #     # Gather both tensor and mask
                #     gathered_tensor = self.accelerator.gather(padded_tensor).cpu()
                #     gathered_mask = self.accelerator.gather(padded_mask).cpu()
                    
                #     # Apply mask to remove padding - only keep valid entries
                #     if gathered_tensor.ndim == 1:
                #         # 1D tensor: simple boolean indexing
                #         clean_tensor = gathered_tensor[gathered_mask]
                #     else:
                #         # Multi-dimensional tensor: mask along first dimension
                #         clean_tensor = gathered_tensor[gathered_mask]
                    
                #     gathered_metrics[key] = clean_tensor
                    
                #     # Log info about padding removal
                #     total_gathered = gathered_tensor.shape[0]
                #     valid_count = clean_tensor.shape[0]
                #     padded_count = total_gathered - valid_count
                #     self.log_debug(f"Gathered metric '{key}': {valid_count} valid, {padded_count} padded entries removed")
                    
                # except Exception as e:
                #     self.accelerator.print(f"Warning: Failed to pad tensor {key} with shape {tensor.shape}, "
                #                         f"attempting direct gather: {e}")
                #     try:
                #         # Fallback: direct gather (assumes no padding needed)
                #         gathered_tensor = self.accelerator.gather(tensor).cpu()
                #         gathered_metrics[key] = gathered_tensor
                #         self.log_debug(f"Gathered metric '{key}' via direct gather: shape {gathered_tensor.shape}")
                #     except Exception as e2:
                #         self.accelerator.print(f"Error: Failed to gather {key}: {e2}")
                #         continue
        
        return gathered_metrics

    def _should_eval(self):
        """
        Determine if evaluation should be run this step.
        
        Returns:
            Boolean indicating whether to run evaluation
            
        Uses eval_every from training_config to decide if current global_step
        warrants an evaluation run. Also makes sure we have a validation dataset
        """
        if self.val_loader is None:
            return False
        if self.training_config.eval_every is None:
            return False
        
        return self.global_step % self.training_config.eval_every == 0
    
    def _tick_early_stopping(self, current_metric: float) -> None:
        """Conduct logic to update early stopping ticker and determine if we should stop."""
        # are we even doing early stopping
        if not self.training_config.patience:
            return False

        # first check if we should even start tracking
        first_step_to_start_tracking = int(self.training_config.early_stopping_sleep_epochs * len(self.train_loader))
        if self.global_step < first_step_to_start_tracking:
            return False
        else:
            
            if self.best_metric is None:
                # here, we asked to start doing early stopping already but we have had no evals yet
                return False
            # alright buddy time to stop slacking
            delta = (current_metric - self.best_metric)
            if self.training_config.primary_metric_mode == 'min':
                delta = -delta
            if delta > self.training_config.min_delta:
                self.evals_no_improve += 1
                self.log_info(f"EARLY STOPPING ticker up to ({self.evals_no_improve}) of  ({self.training_config.patience})"
                              f" observed change since best: {current_metric - self.best_metric}, max allowed: {self.training_config.min_delta}")
                if self.evals_no_improve >= self.training_config.patience:
                    self.log_info(f"EARLY STOPPING triggered at step {self.global_step}")
                    return True
            else:
                self.log_info(f"EARLY STOPPING ticker reset to 0, current change since best: {delta}, max allowed: {self.training_config.min_delta}")

    def _save_checkpoint(
        self, 
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
        # first actually save it
        checkpoint_dir = Path(self.training_config.run_dir) / "checkpoints"
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.global_step}"
        self.accelerator.save_state(checkpoint_path)
        self.log_info(f"Saved checkpoint at step {self.global_step} to {checkpoint_path}")

        # now if best, copy the checkpoint dir to 'best_checkpoint'
        if is_best:
            current_best_checkpoint_path = checkpoint_dir / f"best_checkpoint_{self.global_step}"
            self.accelerator.save_state(current_best_checkpoint_path)
            self.log_info(f"Saved **NEW BEST** checkpoint at step {self.global_step} to {current_best_checkpoint_path}")
        else:
            current_best_checkpoint_path = None

        # now remove previous checkpoints if we are over the limit
        self._cleanup_checkpoints(current_best_checkpoint_path=current_best_checkpoint_path)
        self.accelerator.wait_for_everyone()

    @main_process_only
    def _cleanup_checkpoints(self, current_best_checkpoint_path: Optional[Path] = None) -> None:
        """
        Remove old checkpoints beyond the maximum allowed.
        """
        checkpoint_dir = Path(self.training_config.run_dir) / "checkpoints"
        # first delete the previous best checkpoint if specified
        if current_best_checkpoint_path is not None:
            checkpoints_with_best_tag = list(checkpoint_dir.glob("best_checkpoint_*"))
            for d in checkpoints_with_best_tag:
                # not the current one
                if d != current_best_checkpoint_path:
                    self.log_info(f"Removing old best checkpoint: {d}")
                    shutil.rmtree(d)

        # make sure not to look at the "best" checkpoint
        checkpoint_folders = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")])

        if len(checkpoint_folders) > self.training_config.max_checkpoints:
            checkpoint_steps = [int(d.name.split("_")[-1]) for d in checkpoint_folders]
            steps_to_remove = sorted(checkpoint_steps)[:len(checkpoint_steps) - self.training_config.max_checkpoints]
            for d in checkpoint_folders:
                if int(d.name.split("_")[-1]) in steps_to_remove:
                    self.log_info(f"Removing old checkpoint: {d}")
                    shutil.rmtree(d)

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
        self.accelerator.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=self.global_step)

        log_msg = f"Step {self.global_step}: " + ", ".join([f"{prefix}{k}={v:.4f}" for k, v in metrics.items()])
        self.log_info(log_msg)

    @property
    def best_checkpoint_path_on_file(self):
        checkpoint_dir = Path(self.training_config.run_dir) / "checkpoints"
        best_list = list(checkpoint_dir.glob("best_checkpoint_*"))
        assert len(best_list) <= 1, "Expected at most one best checkpoint"
        if best_list:
            return best_list[0]
        return None

    def _finalize_training(self) -> None:
        """
        Finalize training by saving the last checkpoint and cleaning up resources.
        """
        if self.ema_object is not None:
            self.log_info("Swapping to EMA for final model eval and save")
            self.ema_object.store()
            self.ema_object.copy_to()

        # validate final model and save checkpoint
        # make sure we catch the edge case where this is the best step we update the best checkpoint
        eval_metrics = self._validate()
        
        # determine if is best
        if self.training_config.primary_metric is not None:
            assert self.training_config.primary_metric in eval_metrics, \
                f"Primary metric {self.training_config.primary_metric} not found in eval metrics"
            tracked_eval_metric = eval_metrics[self.training_config.primary_metric]
            is_best = (self.best_metric is None) or (tracked_eval_metric < self.best_metric)
        else:
            is_best = False

        # we need to get the instantanious params back for just a sec for checkpointing
        if self.ema_object is not None:
            self.ema_object.restore()

        # checkpoint
        self._save_checkpoint(is_best=is_best)

        # exit the training interface
        self.accelerator.end_training()

        # we need to save a model with save_pretrained - this differs from a checkpoint eg. it is not wrapped
        # first load the best checkpoint
        if not is_best:
            best_checkpoint_path = self.best_checkpoint_path_on_file
            self.log_info(f"Loading best checkpoint from {best_checkpoint_path}")
            self._load_checkpoint_if_resuming(checkpoint_path=best_checkpoint_path)

        # now we can get the EMA params back for final model save of the best checkpoint
        if self.ema_object is not None:
            self.ema_object.restore()

        # save it - we should be able to call from_pretained on it ...
        self.log_info(f"Saving final model from step={self.global_step}")
        # self.accelerator.save_model(self.model, Path(self.training_config.run_dir) / "final_model")

        # the above is not saving the pretrained config so lets unwrap and try that
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(Path(self.training_config.run_dir) / "final_model")
