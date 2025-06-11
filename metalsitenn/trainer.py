# metalsitenn/trainer.py
'''
* Author: Evan Komp
* Created: 12/11/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

We cannot use a HF default trainer due to the way batches are formatted differently, but
use their nomenclature and functionality when applicable.
'''
from dataclasses import dataclass, asdict, field
import os
import logging
from typing import Optional, Dict, Any, Callable
from datetime import timedelta

import torch 
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
import numpy as np
import time

logger = logging.getLogger(__name__)

def COMPUTE_LOSS_SELF_SUPERVISED_TRAINING(
        trainer: "MetalSiteTrainer",
        input_batch: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
    """Compute the loss for foundational training.

    Returns the total loss and individual losses"""
    out_dict = {}
    assert input_batch['mask'] is not None
    assert input_batch['atom_labels'] is not None
    model_outputs = trainer.model(**input_batch)
    if model_outputs.mask_loss is None:
        raise ValueError("Mask loss is required for foundational training")
    if model_outputs.noise_loss is None:
        raise ValueError("Noise loss is required for foundational training")
    mask_loss = model_outputs.mask_loss
    noise_loss = model_outputs.noise_loss
    total_loss = (1 - trainer.args.frac_noise_loss) * mask_loss + trainer.args.frac_noise_loss * noise_loss

    # normalize by the number of atoms
    out_dict['loss'] = total_loss
    out_dict['mask_loss'] = mask_loss
    out_dict['noise_loss'] = noise_loss

    if return_outputs:
        out_dict['outputs'] = model_outputs

    return out_dict

"""Metrics used during evaluation that require additional computation."""
def NO_MASKING_EMBED_AND_CLUSTER(
        trainer: "MetalSiteTrainer",
        embedding_how: str = "all_mean",
        hidden_state: int = -1,
        cluster_kwargs: Dict[str, Any] = {},
        tsne_kwargs: Dict[str, Any] = {}
    ) -> Dict[str, torch.Tensor]:
    """Compute embeddings using the model, use them to cluster, and see how well clusters pop out.
    
    This metric requires batch manipulation so is called seperately from the eval loop.
    """
    eval_dataloader = trainer.eval_dataloader
    embeddings_all = []
    for batch in eval_dataloader:
        with torch.no_grad():
            model = trainer.accelerator.unwrap_model(trainer.model)

            # update the batch to use actual tokens instead of masks and actual positions
            batch['atoms'] = batch['atom_labels']
            batch['atom_types'] = batch['atom_type_labels']
            batch['pos'] = batch['pos'] + batch['denoise_vectors']

            # get the embeddings
            embeddings = model.embed_systems(
                atoms=batch['atoms'],
                atom_types=batch['atom_types'],
                pos=batch['pos'],
                batch_idx=batch['batch_idx'],
                tokenizer=trainer.data_collator.tokenizer,
                hidden_state=hidden_state,
                how=embedding_how
            )

            gathered_embeddings = trainer.accelerator.gather(embeddings)
            embeddings_all.append(gathered_embeddings.cpu())

    embeddings_all = torch.cat(embeddings_all, dim=0).numpy()

    if trainer.accelerator.is_main_process:

        # cluster the embeddings
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(**cluster_kwargs)

        # fit the clusterer
        clusterer.fit(embeddings_all)

        # get the sillhouette score and log it
        from sklearn.metrics import silhouette_score
        score = silhouette_score(embeddings_all, clusterer.labels_)
        trainer.accelerator.log({"eval/silhouette_score": score}, step=trainer.global_step)

        # lower dimensions using tsne
        from sklearn.manifold import TSNE
        import pandas as pd
        tsne = TSNE(n_components=2, random_state=42, **tsne_kwargs)
        embeddings_2d = tsne.fit_transform(embeddings_all)
        datapoints_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])

        # randomly sample 2000 embeddings if more than that
        if len(datapoints_df) > 2000:
            datapoints_df = datapoints_df.sample(2000, random_state=42)

        # log a plot of the 2d space directly through dvc live
        trainer.accelerator.get_tracker('dvclive', unwrap=True).log_plot(
            "eval_2d_space",
            datapoints_df,
            x='x',
            y='y',
            template='scatter'
        )

        # dvc saves over the same file each run, so if we want a trajectory of these plots we need to save manually
        if not os.path.exists(os.path.join(trainer.args.output_dir, "eval_2d_space")):
            os.makedirs(os.path.join(trainer.args.output_dir, "eval_2d_space"))
        datapoints_df.to_csv(os.path.join(trainer.args.output_dir, "eval_2d_space", f"step_{trainer.global_step}.csv"))
    return

"""Metrics used during evaluation that can be computed directly from model loss outputs."""
def compute_atom_mask_accuracy(trainer, outputs, batch):
    """Accuracy on predicting element"""
    masked_labels = batch['atom_labels'][batch['mask']]
    masked_preds = outputs['outputs'].atom_logits[batch['mask']].argmax(dim=-1)

    return (masked_labels == masked_preds).float().mean().item()

def compute_atom_type_accuracy(trainer, outputs, batch):
    """Accuracy on poredicting hetatm or atm."""
    # check if we are even modeling atom types
    if outputs['outputs'].type_logits is None:
        return 0.0
    masked_labels = batch['atom_type_labels'][batch['mask']]
    masked_preds = outputs['outputs'].type_logits[batch['mask']].argmax(dim=-1)

    return (masked_labels == masked_preds).float().mean().item()

def compute_metal_accuracy(trainer, outputs, batch):
    """Accuracy on specifically predicting the metal token"""
    tokenizer = trainer.data_collator.tokenizer
    metal_token = tokenizer.atom_vocab.metal_token

    masked_labels = batch['atom_labels'][batch['mask']]
    masked_preds = outputs['outputs'].atom_logits[batch['mask']].argmax(dim=-1)

    # get the predictions for only masked tokens that are metals
    metal_mask = masked_labels == metal_token
    if sum(metal_mask) == 0:
        return 0.0
    return (masked_labels[metal_mask] == masked_preds[metal_mask]).float().mean().item()

COMPUTE_EVAL_METRICS_FOUNDATIONAL_TRAINING = {
    'mask_loss': lambda trainer, outputs, batch: outputs['mask_loss'].item(),
    'noise_loss': lambda trainer, outputs, batch: outputs['noise_loss'].item(),
    'atom_mask_accuracy': compute_atom_mask_accuracy,
    'record_mask_accuracy': compute_atom_type_accuracy,
    'metal_mask_accuracy': compute_metal_accuracy
}

@dataclass
class EarlyStoppingState:
    """Tracks early stopping state."""
    counter: int = 0
    best_metric: float = float('inf')
    best_step: int = 0
    
    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.counter = state_dict['counter']
        self.best_metric = state_dict['best_metric']
        self.best_step = state_dict['best_step']

    def step(self, metric: float, current_step: int, min_improvement: float) -> bool:
        """Returns True if should stop."""
        improvement = (self.best_metric - metric) / self.best_metric
        if improvement > min_improvement:
            self.counter = 0
            bad_step =  False
        else:
            bad_step = True
            self.counter += 1
            logger.info(f"Early stopping counter triggered: {self.counter}, best metric: {self.best_metric}, current metric: {metric}, improvement: {improvement}, min improvement: {min_improvement}")
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_step = current_step
        return bad_step

@dataclass
class MetalSiteTrainingArgs:
    """Arguments for training."""
    output_dir: str = field(default="./training_output")
    logging_dir: str = field(default="./logs")
    
    # Training loop
    num_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=8) 
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=0)
    
    # Optimizer
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.0)
    gradient_clipping: float = field(default=1.0)
    warmup_pct: float = field(default=0.1)
    frac_noise_loss: float = field(default=0.5)
    
    # Logging and checkpoints
    eval_steps: int = field(default=None)
    logging_steps: int = field(default=100) 
    load_best_model_at_end: bool = field(default=True)
    
    # Early stopping
    use_early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=3)
    early_stopping_improvement_fraction: float = field(default=0.0)

    def __str__(self):
        return str(asdict(self))

class MetalSiteTrainer:
    """Trainer for metal site models with distributed training support.
    
    Args
    ----
    model: nn.Module
        Model to train
    compute_loss_fn: Callable
        Function to compute loss. Signiture should be:
            compute_loss_fn(trainer: MetalSiteTrainer, input_batch: Dict[str, torch.Tensor], return_outputs: bool = False) -> Dict[str, torch.Tensor]
            Must return dict like with at least a 'loss' key.
            During evaluation, this is called with return_outputs=True to return model outputs for metrics.
    args: MetalSiteTrainingArgs
        Training arguments
    train_dataset: Dataset
        Training dataset
    eval_dataset: Dataset
        Evaluation dataset
    data_collator: Callable
        Data collator
    eval_metrics: Optional[Dict[str, Callable]]
        Metrics to compute during evaluation. This is a dict of callable, each with signature: f(outputs) where outputs are the 
        returns of compute_loss_fn. If None, only loss is computed
    hard_eval_metrics: Optional[Dict[str, Callable]]
        Metrics that require additional computation and are not directly returned by compute_loss_fn. These are called seperately with trainer as the only argument.
        Up to you to loop through whatever dataset to compute it.
    """
    
    def __init__(
        self,
        model,
        compute_loss_fn: Callable,
        args: MetalSiteTrainingArgs,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        eval_metrics: Optional[Dict[str, Callable]]=None,
        hard_eval_metrics: Optional[Dict[str, Callable]]=None,
        quit_early: bool = False,
        resume: bool = False
    ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_loss_fn = compute_loss_fn
        self.eval_metrics = eval_metrics or {}
        
        # Initialize early stopping
        self.early_stopping = EarlyStoppingState() if args.use_early_stopping else None
        
        # Initialize accelerator
        ipgk = InitProcessGroupKwargs(timeout=timedelta(180))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="dvclive",
            project_dir=args.output_dir,
            kwargs_handlers=[ipgk]
        )
        logger.info(f"Accelerator params: {self.accelerator.__dict__}")
        self.accelerator.init_trackers(project_name="training", init_kwargs={
            "dvclive": {
                "dir": os.path.join(args.output_dir, "dvclive"),
                "report": 'md',
                "save_dvc_exp": False,
                "dvcyaml": None,
                'resume': resume
            }
        })
        
        if self.early_stopping:
            self.accelerator.register_for_checkpointing(self.early_stopping)

        # Create dataloaders BEFORE accelerator.prepare()
        self.train_dataloader = self._get_train_dataloader() if train_dataset else None
        self.eval_dataloader = self._get_eval_dataloader() if eval_dataset else None

        # Calculate training schedule BEFORE prepare() using original dataloader length
        self.batches_per_epoch = len(self.train_dataloader)  # Raw batches per epoch (no accumulation, no multi-device)
        self.updates_per_epoch = self.batches_per_epoch // args.gradient_accumulation_steps  # Optimizer updates per epoch (single device)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Create scheduler BEFORE prepare() using single-device update counts
        # Accelerate will handle multi-device scaling internally
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=args.learning_rate,
            epochs=args.num_epochs,
            steps_per_epoch=self.updates_per_epoch,  # Single device updates
            pct_start=args.warmup_pct
        )

        # Prepare everything with accelerator
        prepared = self.accelerator.prepare(
            self.model,
            self.optimizer, 
            self.train_dataloader,
            self.eval_dataloader,
            self.scheduler
        )
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.scheduler = prepared

        # Calculate actual training metrics AFTER prepare() for logging
        self.steps_per_epoch = len(self.train_dataloader)  # Forward passes per epoch (considers multi-device)
        self.total_steps = args.num_epochs * self.steps_per_epoch  # Total forward passes
        self.updates_per_epoch_actual = self.steps_per_epoch // args.gradient_accumulation_steps  # Actual updates per epoch
        self.total_updates = args.num_epochs * self.updates_per_epoch_actual  # Total optimizer updates
        self.n_warmup_updates = int(args.warmup_pct * self.total_updates)  # Warmup in terms of updates

        # Validate eval_steps alignment with gradient accumulation
        if args.eval_steps is not None and args.eval_steps % args.gradient_accumulation_steps != 0:
            raise ValueError(
                f"eval_steps ({args.eval_steps}) must be divisible by gradient_accumulation_steps "
                f"({args.gradient_accumulation_steps}) to ensure evaluation occurs on update boundaries"
            )

        # hard eval metrics
        self.hard_eval_metrics = hard_eval_metrics or {}

        # create checkpointing folder if not present
        if not os.path.exists(os.path.join(args.output_dir, "checkpoints")):
            os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

        self.quit_early = quit_early
        os.environ["NCCL_DEBUG"] = "INFO"

    def _get_train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            shuffle=True
        )

    def _get_eval_dataloader(self) -> DataLoader:
        """Create evaluation dataloader."""
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers
        )
    
    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint with dynamic parameter handling"""
        # Initialize dynamic params before saving
        dummy_batch = next(iter(self.train_dataloader))
        with torch.no_grad():
            self.model(**dummy_batch)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.accelerator.save_state(output_dir, safe_serialization=False)

    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint with dynamic parameter handling"""
        # Initialize dynamic params before loading
        dummy_batch = next(iter(self.train_dataloader))
        with torch.no_grad():
            self.model(**dummy_batch)
            
        self.accelerator.load_state(checkpoint_dir)

    def _cleanup_checkpoints(self):
        """Maintain only best checkpoint and last N checkpoints where N=patience."""
        if not self.early_stopping:
            return
            
        checkpoint_dir = os.path.join(self.args.output_dir, "checkpoints")
        checkpoints = sorted([
            int(f.split('_')[-1]) 
            for f in os.listdir(checkpoint_dir) 
            if f.startswith('step_')
        ])
        
        # Always keep best checkpoint
        checkpoints_to_keep = {self.early_stopping.best_step}
        
        # Keep last patience number of checkpoints
        patience_checkpoints = checkpoints[-self.args.early_stopping_patience:]
        checkpoints_to_keep.update(patience_checkpoints)
        
        # Remove others
        for step in checkpoints:
            if step not in checkpoints_to_keep:
                checkpoint_path = os.path.join(checkpoint_dir, f'step_{step}')
                if os.path.exists(checkpoint_path):
                    import shutil
                    shutil.rmtree(checkpoint_path)

    def evaluate(self) -> float:
        """Run evaluation and compute metrics over full dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Initialize metric accumulators for each process
        process_metrics = {name: [] for name in self.eval_metrics.keys()}
        
        for batch in self.eval_dataloader:
            with torch.no_grad():
                outputs = self.compute_loss_fn(self, batch, return_outputs=True)
                loss = outputs["loss"]
                total_loss += loss.detach().float()
                
                # Compute metrics on each process separately
                if self.eval_metrics:
                    for name, func in self.eval_metrics.items():
                        metric_val = func(self, outputs, batch)
                        if metric_val is not None:
                            process_metrics[name].append(metric_val)
                            
            num_batches += 1

        # Gather and average loss across processes
        total_loss = self.accelerator.gather(total_loss).mean()
        num_batches = self.accelerator.gather(torch.tensor(num_batches, device=self.accelerator.device, dtype=torch.float)).mean()
        avg_loss = total_loss / num_batches

        # Average metrics for each process then gather
        metrics = {"eval/loss": avg_loss.cpu().item()}
        if self.eval_metrics:
            for name, values in process_metrics.items():
                if values:  # Only process if we have values
                    process_avg = torch.tensor(sum(values) / len(values), device=self.accelerator.device)
                    gathered_avgs = self.accelerator.gather(process_avg)
                    metrics[f"eval/{name}"] = gathered_avgs.mean().cpu().item()
                else:
                    metrics[f"eval/{name}"] = float('nan')
                    
        self.accelerator.log(metrics, step=self.global_step)

        # Run any hard metrics
        for name, func in self.hard_eval_metrics.items():
            func(self)
        
        self.model.train()
        torch.cuda.empty_cache()
        return avg_loss.item()

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model."""
        # get memory used on this gpu
        mem = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Memory used on GPU: {mem} GB")
        
        # Initialize global step tracking (forward passes)
        self.global_step = 0
        self.global_update = 0  # Track optimizer updates separately
        start_epoch = 0
        
        if resume_from_checkpoint:
            # Extract step from checkpoint name and calculate positions
            self.global_step = int(resume_from_checkpoint.split('_')[-1])
            self.global_update = self.global_step // self.args.gradient_accumulation_steps
            start_epoch = self.global_update // self.updates_per_epoch_actual
            steps_to_skip = self.global_step % self.steps_per_epoch
            
            self.accelerator.load_state(resume_from_checkpoint)
            
            # Skip batches if resuming mid-epoch
            if steps_to_skip > 0:
                self.train_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader, steps_to_skip
                )
            
            logger.info(f"Resumed from checkpoint: {resume_from_checkpoint} at step {self.global_step}, update {self.global_update}")

        logger.info(
            f"Training with {self.accelerator.num_processes} processes on {self.accelerator.device.type}\n"
            f" - output_dir: {self.args.output_dir}\n"
            f" - examples in dataset: {len(self.train_dataset)}\n"
            f" - per device batch size: {self.args.per_device_train_batch_size}\n"
            f" - gradient accumulation steps: {self.args.gradient_accumulation_steps}\n"
            f" - effective batch size: {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes}\n"
            f" - total epochs: {self.args.num_epochs}\n"
            f" - batches per epoch (raw): {self.batches_per_epoch}\n"
            f" - steps per epoch (forward passes): {self.steps_per_epoch}\n"
            f" - updates per epoch (optimizer steps): {self.updates_per_epoch_actual}\n"
            f" - total steps (forward passes): {self.total_steps}\n"
            f" - total updates (optimizer steps): {self.total_updates}\n"
            f" - warmup updates: {self.n_warmup_updates}\n"
            f" - log training loss every {self.args.logging_steps} steps\n"
            f" - eval and checkpoint every {self.args.eval_steps} steps\n"
            f" - total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # run eval before training
        if not self.quit_early:
            self.evaluate()
        
        # Training loop
        for epoch in range(start_epoch, self.args.num_epochs):
            self.model.train()
            total_loss = 0
            loss_accumulation_count = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                disable=not self.accelerator.is_local_main_process,
                total=len(self.train_dataloader)
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    logger.debug(f"N nodes in graph: {len(batch['atoms'])}")
                    outputs = self.compute_loss_fn(self, batch)
                    loss = outputs["loss"]
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        # This happens every gradient_accumulation_steps batches
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.gradient_clipping
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        
                        # Increment update counter on optimizer updates
                        self.global_update += 1

                        if self.quit_early:
                            logger.info("Quitting early")
                            return

                    # Always increment step counter on forward passes
                    self.global_step += 1

                    # Accumulate loss for logging
                    total_loss += loss.detach().float()
                    loss_accumulation_count += 1

                # Log training metrics
                if (
                    self.global_step > 0 
                    and self.global_step % self.args.logging_steps == 0
                ):
                    avg_loss = total_loss / loss_accumulation_count
                    self.accelerator.log({
                        "train/loss": avg_loss.item(),
                        "train/epoch": epoch,
                        "train/global_step": self.global_step,
                        "train/global_update": self.global_update,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"]
                    }, step=self.global_step)
                    total_loss = 0
                    loss_accumulation_count = 0

                # Evaluate and checkpoint if needed (only on update boundaries)
                if (
                    self.args.eval_steps 
                    and self.global_step > 0 
                    and self.global_step % self.args.eval_steps == 0
                ):
                    eval_loss = self.evaluate()
                    self.model.train()

                    # Save checkpoint
                    output_dir = os.path.join(
                        self.args.output_dir,
                        "checkpoints",
                        f"step_{self.global_step}"
                    )
                    # Ensure the directory exists on all processes before saving
                    if self.accelerator.is_main_process:
                        os.makedirs(output_dir, exist_ok=True)
                    self.accelerator.wait_for_everyone()  # Wait for directory creation to complete
                    self.save_checkpoint(output_dir)
                    if self.accelerator.is_main_process:
                        self._cleanup_checkpoints()

                    if self.early_stopping:
                        should_stop = self.early_stopping.step(
                            eval_loss,
                            self.global_step,
                            self.args.early_stopping_improvement_fraction
                        )
                        if (should_stop and 
                            self.early_stopping.counter >= self.args.early_stopping_patience):
                            if self.global_update > self.n_warmup_updates:
                                logger.info("Early stopping triggered")
                                self._finish_up()
                                return

        self._finish_up()

    def _finish_up(self):
        """Save final checkpoint and load best model if requested."""
        output_dir = os.path.join(
            self.args.output_dir,
            "checkpoints",
            f"step_{self.global_step}"
        )
        self.save_checkpoint(output_dir)

        if self.args.load_best_model_at_end and self.early_stopping and self.early_stopping.best_step > 0:
            best_model_path = os.path.join(
                self.args.output_dir,
                "checkpoints",
                f"step_{self.early_stopping.best_step}"
            )
            logger.info(f"Loading best model from step {self.early_stopping.best_step}")
            self.load_checkpoint(best_model_path)