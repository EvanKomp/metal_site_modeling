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
    period: int = 100
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
        elif lambda_type == "multistep_cosine":
            lambda_fn = _create_multistep_cosine_lambda(total_steps, warmup_steps, warmup_factor, lr_min_factor, period)
        elif lambda_type == "warm_restart_cosine":
            lambda_fn = _create_warm_restart_cosine_lambda(total_steps, warmup_steps, warmup_factor, lr_min_factor, period)
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


def _create_multistep_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    warmup_factor: float,
    lr_min_factor: float,
    period: int
) -> callable:
    """
    Create multistep cosine lambda with symmetric sine cycles after warmup.
    
    After warmup, creates symmetric sine curves that start at 1.0, go down to 
    lr_min_factor at period/2, then back up to 1.0 at period. Each cycle is 
    continuous with no jumps.
    
    Args:
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        warmup_factor: Starting factor for warmup phase
        lr_min_factor: Minimum LR factor for sine cycles
        period: Steps per complete sine cycle (down and back up)
        
    Returns:
        Lambda function for LambdaLR scheduler
    """
    def multistep_cosine_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            alpha = step / max(warmup_steps, 1)
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            # Symmetric sine cycles: starts at 1.0, goes to lr_min_factor, back to 1.0
            steps_since_warmup = step - warmup_steps
            cycle_progress = (steps_since_warmup % period) / max(period, 1)
            cycle_progress = min(cycle_progress, 1.0)
            
            # Use cosine to create symmetric curve: cos(0)=1, cos(π)=-1, cos(π/2)=0
            # Map to desired range: 1.0 at start/end, lr_min_factor at middle
            cosine_value = math.cos(2 * math.pi * cycle_progress)
            # Remap from [-1,1] to [lr_min_factor, 1.0]
            return lr_min_factor + 0.5 * (1.0 - lr_min_factor) * (1.0 + cosine_value)
    
    return multistep_cosine_lambda

def _create_warm_restart_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    warmup_factor: float,
    lr_min_factor: float,
    period: int
) -> callable:
    """
    Create warm restart cosine lambda with periodic restarts after warmup.
    
    After warmup, each period starts at 1.0 and decays via cosine to lr_min_factor,
    then jumps back to 1.0 for the next period. Classic cosine annealing with
    warm restarts (SGDR) pattern.
    
    Args:
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        warmup_factor: Starting factor for warmup phase
        lr_min_factor: Minimum LR factor for cosine decay
        period: Steps per cosine decay cycle
        
    Returns:
        Lambda function for LambdaLR scheduler
    """
    def warm_restart_cosine_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            alpha = step / max(warmup_steps, 1)
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            # Cosine annealing with warm restarts
            steps_since_warmup = step - warmup_steps
            cycle_progress = (steps_since_warmup % period) / max(period, 1)
            cycle_progress = min(cycle_progress, 1.0)
            
            # Standard cosine decay: cos(0)=1 → cos(π)=-1
            # Maps to: 1.0 at start → lr_min_factor at end of period
            return lr_min_factor + 0.5 * (1.0 - lr_min_factor) * (
                1.0 + math.cos(math.pi * cycle_progress)
            )
 
    return warm_restart_cosine_lambda
