# metalsitenn/trainer.py
'''
* Author: Evan Komp
* Created: 12/11/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

We cannot use a HF default trainer due to the way batches are formatted differently, but
use their nomenclature and functionality when applicable.
'''
from dataclasses import dataclass, field, asdict
import json
import os

from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from typing import Optional, List, Dict, Callable

from metalsitenn.model import ModelOutput, MetalSitePretrainedModel

import logging
logger = logging.getLogger(__name__)

@dataclass(frozen=False)
class MetalSiteTrainingArgs:
    """Arguments for training a MetalSite model.
    
    Follows the huggingface nomenclature and functionality.
    """
    # pathing
    output_dir: str = field(default="./training_output", metadata={"help": "Directory to save model checkpoints and logs."})
    logging_file: str = field(default="training.log", metadata={"help": "File to save training logs."})
    # logging and evaluation
    eval_steps: int = field(default=None, metadata={"help": "Number of steps between evaluation and save runs."})
    logging_steps: int = field(default=100, metadata={"help": "Number of steps between logging."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load the best model at the end of training."})
    # training batches and steps
    num_epochs: int = field(default=1, metadata={"help": "Number of epochs to train for."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for evaluation."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients over."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of workers for dataloader."})
    # optimizer
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate for optimizer (Adam)."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for optimizer."})
    gradient_clipping: float = field(default=1.0, metadata={"help": "Gradient clipping value."})
    frac_noise_loss: float = field(default=0.5, metadata={"help": "Fraction of total loss allocated to denoising, the remainder given to demasking."})
    # scheduler
    warmup_steps: int = field(default=0, metadata={"help": "Number of steps for linear warmup."})
    # early stopping
    use_early_stopping: bool = field(default=False, metadata={"help": "Use early stopping."})
    early_stopping_patience: int = field(default=3, metadata={"help": "Number of evaluations without improvement before stopping."})
    early_stopping_improvement_fraction: float = field(default=0.0, metadata={"help": "Fraction of improvement needed to continue training."})
    
@dataclass
class TrainerState:
    """State of the trainer."""
    global_step: int
    epoch: int
    early_stopping_patience_counter: int
    metric: float=None
    best_metric: float=field(default=float('inf'))

    def save(self, output_dir: str):
        """Save the state to a file."""
        with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, output_dir: str):
        """Load the state from a file."""
        with open(os.path.join(output_dir, "trainer_state.json"), "r") as f:
            state_dict = json.load(f)
        return cls(**state_dict)


class MetalSiteTrainer:
    """Trainer for MetalSiteNN models that handles distributed training via Accelerate.
    
    Coordinates:
    - Model training and evaluation loops
    - Distributed training across GPUs/machines 
    - Checkpointing and model saving
    - Metric logging and early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model,
        compute_loss_fn: Callable,
        args: MetalSiteTrainingArgs,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
    ):
        """Initialize trainer instance.
        
        Args:
            model: Model to train
            args: Training arguments defining hyperparameters and settings
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset 
            data_collator: Optional custom batch collation function
        """
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_loss_fn = compute_loss_fn
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.gradient_clipping,
        )
        # get the number of devices from accelerator
        self._num_devices = self.accelerator.num_processes

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  
            level=logging.INFO,
            handlers=[logging.FileHandler(args.logging_file)]
        )
        self.logger = logging.getLogger(__name__)

        # Create dataloaders
        self.train_dataloader = self._get_train_dataloader() if train_dataset is not None else None
        self.eval_dataloader = self._get_eval_dataloader() if eval_dataset is not None else None

        # Set up optimizer and scheduler 
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()

        # Prepare with accelerator
        prepared = self.accelerator.prepare(
            self.model,
            self.optimizer, 
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler
        )
        self.model = prepared[0]
        self.optimizer = prepared[1] 
        self.train_dataloader = prepared[2]
        self.eval_dataloader = prepared[3]
        self.lr_scheduler = prepared[4]

        self.state = TrainerState(
            global_step=0,
            epoch=0,
            metric=None,
            best_metric=float('inf'),
            early_stopping_patience_counter=0
        )

    def _get_train_dataloader(self) -> DataLoader:
        """Create the training dataloader.
        
        Returns:
            DataLoader for training dataset with appropriate batch size and collation
        """         
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            shuffle=True
        )
        logger.info(f"Created training dataloader with {len(dl)} batches.")


    def _get_eval_dataloader(self) -> DataLoader:
        """Create the evaluation dataloader.
        
        Returns:
            DataLoader for evaluation dataset with appropriate batch size and collation
        """
        dl = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers
        )
        logger.info(f"Created evaluation dataloader with {len(dl)} batches.")
        return dl

    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler.
        
        Returns:
            tuple: (optimizer, scheduler)
        """
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Calculate key training quantities
        samples_per_step = (
            self.args.per_device_train_batch_size * 
            self._num_devices * 
            self.args.gradient_accumulation_steps
        )
        steps_per_epoch = len(self.train_dataset) // samples_per_step
        total_training_steps = steps_per_epoch * self.args.num_epochs
        
        # Store for training loop
        self.num_training_steps = total_training_steps
        
        # Create scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_training_steps
        )

        # Log all quantities
        logger.info(
            f"\nTraining setup:"
            f"\n- Total dataset size: {len(self.train_dataset)}"
            f"\n- Number of devices: {self._num_devices}"
            f"\n- Per device batch size: {self.args.per_device_train_batch_size}"
            f"\n- Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
            f"\n- Effective batch size per step: {samples_per_step}"
            f"\n- Steps per epoch: {steps_per_epoch}"
            f"\n- Number of epochs: {self.args.num_epochs}"
            f"\n- Total training steps: {total_training_steps}"
            f"\n- Warmup steps: {self.args.warmup_steps}"
        )
        
        return optimizer, scheduler
    
    def _get_num_atoms_in_batch(self, batch):
        """Get the number of atoms in a batch.
        
        Args:
            batch: Dictionary of batch data
        
        Returns:
            int: Number of atoms in the batch
        """
        return len(batch['atoms'])
    
    def compute_loss(self, input_batch: Dict[str, torch.Tensor], return_outputs: bool = False) -> Dict[str, torch.Tensor]:
        """Compute the loss for a training step.
        
        Args:
            input_batch: Dictionary of input data
            return_outputs: Whether to return model outputs
        
        Returns:
            Dictionary of losses and optionally model outputs
        """
        return self.compute_loss_fn(self, input_batch, return_outputs)

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model with multi-GPU support via Accelerate."""

        # Calculate true number of total steps accounting for all devices
        num_update_steps_per_epoch = (
            len(self.train_dataloader) // self.args.gradient_accumulation_steps
        ) 
        num_total_steps = num_update_steps_per_epoch * self.args.num_epochs

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")

        self.logger.info(
            f"Training setup:\n"
            f"- Total steps: {num_total_steps}\n"
            f"- Steps per epoch: {num_update_steps_per_epoch}\n" 
            f"- Gradient accumulation steps: {self.args.gradient_accumulation_steps}\n"
            f"- Number of devices: {self._num_devices}"
        )

        # Training loop
        for epoch in range(self.state.epoch, self.args.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = enumerate(self.train_dataloader)
            if self.accelerator.is_main_process:
                progress_bar = tqdm(progress_bar, total=len(self.train_dataloader))

            for step, batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    outputs = self.compute_loss(batch)
                    loss = outputs["loss"]
                    
                    # Scale loss by number of atoms and devices
                    num_atoms = self._get_num_atoms_in_batch(batch) * self._num_devices
                    loss = loss / num_atoms
                    
                    self.accelerator.backward(loss)
                    
                    # Only update on sync steps
                    if not self.accelerator.sync_gradients:
                        continue
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.state.global_step += 1

                    # Logging (only on main process)
                    if self.accelerator.is_main_process:
                        if self.state.global_step % self.args.logging_steps == 0:
                            total_loss = self.accelerator.gather(total_loss).mean()
                            avg_loss = total_loss / self.args.logging_steps
                            self.logger.info(
                                f"Epoch: {epoch}, Step: {self.state.global_step}, "
                                f"Loss: {avg_loss:.4f}, LR: {self.lr_scheduler.get_last_lr()[0]:.2e}"
                            )
                            total_loss = 0

                        # Evaluate and save if needed 
                        if self.args.eval_steps and self.state.global_step % self.args.eval_steps == 0:
                            eval_loss = self.evaluate()
                            self.model.train()

                            if eval_loss < self.state.best_metric:
                                self.state.best_metric = eval_loss
                                self._checkpoint(os.path.join(self.args.output_dir, "best"))
                            
                            if self.args.use_early_stopping:
                                self._check_early_stopping(eval_loss)
                                if self.state.early_stopping_patience_counter >= self.args.early_stopping_patience:
                                    self.logger.info("Early stopping triggered")
                                    return

            self.state.epoch += 1
            
            # Save epoch checkpoint on main process
            if self.accelerator.is_main_process:
                self._checkpoint(os.path.join(self.args.output_dir, f"epoch_{epoch}"))

        # Load best model if requested
        if self.args.load_best_model_at_end and self.accelerator.is_main_process:
            self._load_checkpoint(os.path.join(self.args.output_dir, "best"))

    def _check_early_stopping(self, eval_loss: float):
        """Check if early stopping criteria are met."""
        improvement = (self.best_metric - eval_loss) / self.best_metric
        if improvement < self.args.early_stopping_improvement_fraction:
            self.early_stopping_patience_counter += 1
        else:
            self.early_stopping_patience_counter = 0

    def _checkpoint(self, location: str):
        """Save a checkpoint."""
        self.accelerator.save_state(os.path.join(location, "acc_checkpoint"))
        self.state.save(location)

    def _load_checkpoint(self, location: str):
        """Load a checkpoint."""
        self.accelerator.load_state(os.path.join(location, "acc_checkpoint"))
        self.state = TrainerState.load(location)

    
def compute_loss_foundational_training_loss(
        self,
        trainer: MetalSiteTrainer,
        input_batch: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
    """Compute the loss for foundational training.

    Returns the total loss and individual losses"""
    out_dict = {}
    model_outputs = trainer.model(**input_batch)
    mask_loss = model_outputs.mask_loss if model_outputs.mask_loss is not None else 0
    noise_loss = model_outputs.noise_loss if model_outputs.noise_loss is not None else 0
    total_loss = (1 - trainer.args.frac_noise_loss) * mask_loss + trainer.args.frac_noise_loss * noise_loss

    # normalize by the number of atoms
    out_dict['loss'] = total_loss
    out_dict['mask_loss'] = mask_loss
    out_dict['noise_loss'] = noise_loss

    if return_outputs:
        out_dict['outputs'] = model_outputs

    return out_dict
        
