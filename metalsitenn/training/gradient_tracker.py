# metalsitenn/training/gradient_tracker.py
'''
* Author: Evan Komp
* Created: 9/1/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


class GradientTracker:
    """
    Tracks gradient statistics during training for debugging purposes.
    
    Uses enhanced pattern matching with '#' and '*' operators for flexible parameter grouping:
    - '*' (group all): Groups all matching parameters together
    - '#' (separate each): Creates separate groups for each matching submodule
    
    Pattern Examples:
    - 'mlp.network.*' or 'mlp.network': Groups all params in mlp.network together
    - 'mlp.network.#': Creates separate groups for each submodule in mlp.network
    - 'mlp.*.weight': Groups all weight parameters across mlp submodules
    - 'mlp.network.#.*.weight': For each submodule in mlp.network, group all weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        track_patterns: Optional[Union[List[str], Dict[str, str]]] = None,
        track_flow: bool = True
    ):
        """
        Initialize gradient tracker with enhanced pattern matching.
        
        Args:
            model: PyTorch model to track gradients for
            track_patterns: Pattern specification using '#' and '*' operators
            track_flow: Whether to track gradient flow through network depth
        """
        self.model = model
        self.track_patterns = track_patterns
        self.track_flow = track_flow
        
        # Internal state
        self._layer_groups: Dict[str, List[str]] = {}
        self._setup_layer_tracking()

    def _get_unwrapped_model(self) -> nn.Module:
        """
        Get the underlying model, unwrapping DDP/FSDP/other wrappers.
        
        Returns:
            The unwrapped PyTorch model
        """
        model = self.model
        
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model = model.module
            
        # Handle other common wrappers (FSDP, etc.)
        while hasattr(model, '_orig_mod'):
            model = model._orig_mod
            
        return model
        
    def _setup_layer_tracking(self) -> None:
        """Setup tracking groups based on enhanced patterns."""
        # Handle DDP wrapped models by accessing the underlying module
        model_for_params = self._get_unwrapped_model()
        
        all_named_params = [(name, param) for name, param in model_for_params.named_parameters() 
                        if param.requires_grad]
        
        if self.track_patterns is None:
            # Track all parameters individually
            self._layer_groups = {name: [name] for name, _ in all_named_params}
            
        elif isinstance(self.track_patterns, dict):
            # Custom group names with patterns
            self._layer_groups = self._group_by_custom_patterns(
                self.track_patterns, all_named_params
            )
            
        elif isinstance(self.track_patterns, list):
            # List of patterns - auto-generate group names
            self._layer_groups = self._group_by_pattern_list(
                self.track_patterns, all_named_params
            )
                
        logger.info(f"Tracking {len(self._layer_groups)} groups: {list(self._layer_groups.keys())}")
    
    def _group_by_custom_patterns(
        self, 
        pattern_dict: Dict[str, str], 
        all_named_params: List[tuple]
    ) -> Dict[str, List[str]]:
        """Group parameters using custom pattern dictionary."""
        groups = defaultdict(list)
        available_names = {name for name, _ in all_named_params}
        
        for group_name, pattern in pattern_dict.items():
            matched_groups = self._match_pattern_to_groups(pattern, available_names)
            for match_key, param_list in matched_groups.items():
                if param_list:
                    # Use custom group name if no separators, otherwise append match
                    if '#' in pattern:
                        final_group_name = f"{group_name}.{match_key.split('.')[-1]}"
                    else:
                        final_group_name = group_name
                    groups[final_group_name] = param_list
        
        return dict(groups)
    
    def _group_by_pattern_list(
        self, 
        patterns: List[str], 
        all_named_params: List[tuple]
    ) -> Dict[str, List[str]]:
        """Group parameters using list of patterns with auto-generated names."""
        groups = defaultdict(list)
        available_names = {name for name, _ in all_named_params}
        
        for pattern in patterns:
            matched_groups = self._match_pattern_to_groups(pattern, available_names)
            for group_key, param_list in matched_groups.items():
                if param_list:
                    groups[group_key] = param_list
        
        return dict(groups)
    
    def _match_pattern_to_groups(
        self, 
        pattern: str, 
        available_names: set
    ) -> Dict[str, List[str]]:
        """
        Match an enhanced pattern to parameter names and group them.
        
        Handles both '#' (separate) and '*' (group all) operators.
        
        Args:
            pattern: Enhanced pattern like 'mlp.network.#' or 'mlp.*.weight'
            available_names: Set of all available parameter names
            
        Returns:
            Dict mapping group keys to lists of matching parameter names
        """
        groups = defaultdict(list)
        
        # Normalize pattern - treat 'mlp.network.*' same as 'mlp.network'
        if pattern.endswith('.*'):
            pattern = pattern[:-2]
        
        # Handle different pattern types
        if '#' not in pattern and '*' not in pattern:
            # Simple prefix pattern - group all together
            matched_params = [name for name in available_names 
                            if name.startswith(pattern)]
            if matched_params:
                groups[pattern] = matched_params
                
        elif '#' in pattern and '*' not in pattern:
            # Separate submodules pattern: 'mlp.network.#'
            groups.update(self._handle_separate_pattern(pattern, available_names))
            
        elif '*' in pattern and '#' not in pattern:
            # Group all pattern with wildcards: 'mlp.*.weight'
            groups.update(self._handle_group_all_pattern(pattern, available_names))
            
        else:
            # Mixed pattern: 'mlp.network.#.*.weight'
            groups.update(self._handle_mixed_pattern(pattern, available_names))
        
        return dict(groups)
    
    def _handle_separate_pattern(self, pattern: str, available_names: set) -> Dict[str, List[str]]:
        """
        Handle patterns with '#' operator for separating submodules.
        
        Example: 'mlp.network.#' finds all submodules in mlp.network and creates
        separate groups for each one.
        """
        groups = defaultdict(list)
        
        # Remove '#' and get the base pattern
        base_pattern = pattern.replace('#', '')
        if base_pattern.endswith('.'):
            base_pattern = base_pattern[:-1]
        
        # Find all submodules at the specified level
        submodules = set()
        for param_name in available_names:
            if param_name.startswith(base_pattern + '.'):
                # Extract the immediate submodule name
                remainder = param_name[len(base_pattern) + 1:]
                if '.' in remainder:
                    submodule = remainder.split('.')[0]
                    submodules.add(f"{base_pattern}.{submodule}")
        
        # Group parameters by submodule
        for submodule in submodules:
            matched_params = [name for name in available_names 
                            if name.startswith(submodule)]
            if matched_params:
                groups[submodule] = matched_params
        
        return dict(groups)
    
    def _handle_group_all_pattern(self, pattern: str, available_names: set) -> Dict[str, List[str]]:
        """
        Handle patterns with '*' operator for grouping all matches.
        
        Example: 'mlp.*.weight' finds all weight parameters in any mlp submodule
        and groups them together.
        """
        groups = defaultdict(list)
        
        # Convert pattern to regex
        # 'mlp.*.weight' -> 'mlp\\.([^.]+)\\.weight'
        regex_pattern = self._pattern_to_regex(pattern, group_wildcards=False)
        compiled_pattern = re.compile(regex_pattern)
        
        matched_params = []
        for param_name in available_names:
            if compiled_pattern.match(param_name):
                matched_params.append(param_name)
        
        if matched_params:
            # Use pattern as group key, removing '*' for cleaner names
            group_key = pattern.replace('.*', '').replace('*', 'all')
            groups[group_key] = matched_params
        
        return dict(groups)
    
    def _handle_mixed_pattern(self, pattern: str, available_names: set) -> Dict[str, List[str]]:
        """
        Handle patterns with both '#' and '*' operators.
        
        Example: 'mlp.network.#.*.weight' separates each submodule in mlp.network,
        then groups all weight parameters within each submodule.
        """
        groups = defaultdict(list)
        
        # Split pattern at '#' to get base and suffix
        parts = pattern.split('#')
        if len(parts) != 2:
            logger.warning(f"Invalid mixed pattern: {pattern}")
            return dict(groups)
        
        base_pattern = parts[0].rstrip('.')
        suffix_pattern = parts[1].lstrip('.')
        
        # First find all submodules using the base pattern
        submodule_groups = self._handle_separate_pattern(base_pattern + '#', available_names)
        
        # Then apply suffix pattern within each submodule
        for submodule_key, submodule_params in submodule_groups.items():
            submodule_names = set(submodule_params)
            
            # Apply suffix pattern to this submodule's parameters
            if '*' in suffix_pattern:
                # Build full pattern for this submodule
                full_pattern = f"{submodule_key}.{suffix_pattern}"
                regex_pattern = self._pattern_to_regex(full_pattern, group_wildcards=False)
                compiled_pattern = re.compile(regex_pattern)
                
                matched_params = [name for name in submodule_names 
                                if compiled_pattern.match(name)]
                if matched_params:
                    # Create group key combining submodule and suffix
                    suffix_clean = suffix_pattern.replace('.*', '').replace('*', 'all')
                    group_key = f"{submodule_key}.{suffix_clean}"
                    groups[group_key] = matched_params
            else:
                # Simple suffix - direct prefix match
                full_prefix = f"{submodule_key}.{suffix_pattern}"
                matched_params = [name for name in submodule_names 
                                if name.startswith(full_prefix)]
                if matched_params:
                    groups[full_prefix] = matched_params
        
        return dict(groups)
    
    def _pattern_to_regex(self, pattern: str, group_wildcards: bool = True) -> str:
        """
        Convert enhanced pattern to regex.
        
        Args:
            pattern: Pattern with '*' wildcards  
            group_wildcards: If True, capture wildcards in groups
            
        Returns:
            Regex pattern string
        """
        # Escape dots
        regex_pattern = pattern.replace('.', r'\.')
        
        # Handle wildcards - match any sequence of path segments
        if group_wildcards:
            regex_pattern = regex_pattern.replace('*', r'([^.]+(?:\.[^.]+)*)')
        else:
            regex_pattern = regex_pattern.replace('*', r'[^.]+(?:\.[^.]+)*')
        
        # Exact match for parameter names
        regex_pattern = f"^{regex_pattern}$"
        
        return regex_pattern
    
    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Compute gradient statistics for current model state.
        
        Returns:
            Dictionary of gradient metrics as tensors ready for logging
        """
        if not self._has_gradients():
            return {}
            
        metrics = {}
        
        # Core gradient metrics
        metrics.update(self._compute_grouped_gradient_norms())
        metrics.update(self._compute_grouped_parameter_ratios())
        
        if self.track_flow:
            metrics.update(self._compute_gradient_flow())
            
        return metrics
    
    def _has_gradients(self) -> bool:
        """Check if model currently has gradients."""
        for _, param in self.model.named_parameters():
            if param.grad is not None:
                return True
        return False
    
    def _compute_grouped_gradient_norms(self) -> Dict[str, torch.Tensor]:
        """Compute L2 norms of gradients for tracked layer groups."""
        metrics = {}
        # Use unwrapped model for parameter access
        param_dict = dict(self._get_unwrapped_model().named_parameters())
        total_norm_sq = 0.0
        group_count = 0
        
        for group_name, param_names in self._layer_groups.items():
            group_norm_sq = 0.0
            param_count = 0
            
            for param_name in param_names:
                param = param_dict.get(param_name)
                if param is not None and param.grad is not None:
                    grad_norm_sq = param.grad.data.norm(2).item() ** 2
                    group_norm_sq += grad_norm_sq
                    param_count += 1
            
            if param_count > 0:
                group_norm = torch.tensor(group_norm_sq ** 0.5, 
                                        device=next(self.model.parameters()).device)
                metrics[f"grad_norm/{group_name}"] = group_norm.unsqueeze(0)
                total_norm_sq += group_norm_sq
                group_count += 1
        
        # Global gradient norm
        if group_count > 0:
            total_norm = torch.tensor(total_norm_sq ** 0.5, 
                                    device=next(self.model.parameters()).device)
            metrics["grad_norm/total"] = total_norm.unsqueeze(0)
            
            avg_norm = torch.tensor((total_norm_sq / group_count) ** 0.5, 
                                device=next(self.model.parameters()).device)
            metrics["grad_norm/average"] = avg_norm.unsqueeze(0)
            
        return metrics

    def _compute_grouped_parameter_ratios(self) -> Dict[str, torch.Tensor]:
        """Compute gradient-to-parameter magnitude ratios for groups."""
        metrics = {}
        # Use unwrapped model for parameter access  
        param_dict = dict(self._get_unwrapped_model().named_parameters())
        
        for group_name, param_names in self._layer_groups.items():
            group_grad_norm_sq = 0.0
            group_param_norm_sq = 0.0
            
            for param_name in param_names:
                param = param_dict.get(param_name)
                if param is not None and param.grad is not None:
                    group_grad_norm_sq += param.grad.data.norm(2).item() ** 2
                    group_param_norm_sq += param.data.norm(2).item() ** 2
            
            if group_param_norm_sq > 1e-8:
                group_grad_norm = group_grad_norm_sq ** 0.5
                group_param_norm = group_param_norm_sq ** 0.5
                ratio = torch.tensor(group_grad_norm / group_param_norm, 
                                device=next(self.model.parameters()).device)
                metrics[f"grad_param_ratio/{group_name}"] = ratio.unsqueeze(0)
                
        return metrics
    
    def _compute_gradient_flow(self) -> Dict[str, torch.Tensor]:
        """Compute gradient flow metrics through network depth."""
        metrics = {}
        group_norms = []
        
        # Collect gradient norms by group order
        for group_name, param_names in self._layer_groups.items():
            param_dict = dict(self.model.named_parameters())
            group_norm_sq = 0.0
            
            for param_name in param_names:
                param = param_dict.get(param_name)
                if param is not None and param.grad is not None:
                    group_norm_sq += param.grad.data.norm(2).item() ** 2
            
            if group_norm_sq > 0:
                group_norms.append(group_norm_sq ** 0.5)
        
        if len(group_norms) > 1:
            device = next(self.model.parameters()).device
            
            # Gradient flow variance (evenness of gradient distribution)
            flow_variance = torch.var(torch.tensor(group_norms, device=device))
            metrics["grad_flow/variance"] = flow_variance.unsqueeze(0)
            
            # Gradient flow ratio (first to last group)
            flow_ratio = torch.tensor(group_norms[0] / (group_norms[-1] + 1e-8), device=device)
            metrics["grad_flow/first_to_last_ratio"] = flow_ratio.unsqueeze(0)
            
        return metrics
    
    def get_tracked_groups(self) -> Dict[str, List[str]]:
        """Return dictionary of currently tracked layer groups."""
        return self._layer_groups.copy()
    
    def update_tracking_config(
        self, 
        track_patterns: Optional[Union[List[str], Dict[str, str]]] = None,
        track_flow: Optional[bool] = None
    ) -> None:
        """Update tracking configuration dynamically."""
        if track_patterns is not None:
            self.track_patterns = track_patterns
            self._setup_layer_tracking()
            
        if track_flow is not None:
            self.track_flow = track_flow