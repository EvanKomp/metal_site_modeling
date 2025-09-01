# metalsitenn/training/lr_scheduler_factory.py
'''
* Author: Evan Komp
* Created: 9/1/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import math
from typing import Optional, List
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_factor: float = 0.2,
    lr_min_factor: float = 0.01,
    decay_epochs: Optional[List[int]] = None,
    decay_rate: float = 0.1,
    lambda_type: str = "cosine",
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from configuration parameters.
    
    Args:
        optimizer: PyTorch optimizer instance
        scheduler_type: Type of scheduler ("LambdaLR", "CosineAnnealingLR", "StepLR", "None")
        total_steps: Total training steps for the schedule
        warmup_steps: Number of warmup steps (linear ramp-up)
        warmup_factor: Starting factor for warmup phase
        lr_min_factor: Minimum LR factor for cosine annealing
        decay_epochs: List of epoch indices for multistep decay
        decay_rate: Decay factor for multistep/step schedulers
        lambda_type: Lambda function type for LambdaLR ("cosine" or "multistep")
        
    Returns:
        Configured scheduler instance or None if scheduler_type is "None"
        
    Raises:
        ValueError: For invalid scheduler configurations
    """
    if scheduler_type == "None":
        return None
        
    if scheduler_type == "LambdaLR":
        if lambda_type == "cosine":
            lambda_fn = _create_cosine_lambda(total_steps, warmup_steps, warmup_factor, lr_min_factor)
        elif lambda_type == "multistep":
            if decay_epochs is None:
                raise ValueError("decay_epochs required for multistep lambda scheduler")
            lambda_fn = _create_multistep_lambda(warmup_steps, warmup_factor, decay_epochs, decay_rate, total_steps)
        else:
            raise ValueError(f"Unsupported lambda_type: {lambda_type}")
            
        return LambdaLR(optimizer, lambda_fn)
        
    elif scheduler_type == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=optimizer.param_groups[0]['lr'] * lr_min_factor
        )
        
    elif scheduler_type == "StepLR":
        # Default to 10-epoch steps if not specified
        step_size = total_steps // 10 if total_steps > 10 else 1
        return StepLR(optimizer, step_size=step_size, gamma=decay_rate)
        
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def _create_cosine_lambda(
    total_steps: int, 
    warmup_steps: int, 
    warmup_factor: float, 
    lr_min_factor: float
) -> callable:
    """Create cosine annealing lambda function with warmup."""
    def cosine_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            alpha = step / max(warmup_steps, 1)
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            return lr_min_factor + 0.5 * (1.0 - lr_min_factor) * (
                1.0 + math.cos(math.pi * progress)
            )
    
    return cosine_lambda


def _create_multistep_lambda(
    warmup_steps: int,
    warmup_factor: float, 
    decay_epochs: List[int],
    decay_rate: float,
    total_steps: int
) -> callable:
    """Create multistep lambda function with warmup."""
    # Convert decay epochs to steps (assumes uniform epoch length)
    steps_per_epoch = total_steps // max(decay_epochs[-1], 1) if decay_epochs else 1
    decay_steps = [epoch * steps_per_epoch for epoch in decay_epochs]
    
    def multistep_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            alpha = step / max(warmup_steps, 1)
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            # Multistep decay
            decay_count = sum(1 for decay_step in decay_steps if step >= decay_step)
            return decay_rate ** decay_count
    
    return multistep_lambda