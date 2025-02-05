# metalsitenn/utils.py
'''
* Author: Evan Komp
* Created: 11/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pandas as pd
from dataclasses import dataclass
import re
from typing import List, Tuple, Iterator, Dict, Any
import torch

@dataclass
class ParamsObj:
    """Wraps dict of dicts to allow attribute access to nested dicts."""
    def __init__(self, upper_dict):
        for k, v in upper_dict.items():
            if isinstance(v, dict):
                setattr(self, k, ParamsObj(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return str(self.__dict__)
    
    @property
    def param_names(self):
        return list(self.__dict__.keys())
    
def get_emission_time_job_from_codecarbon_log(emissions_file: str, project_name: str) -> pd.DataFrame:
    """
    Parse the CodeCarbon emissions log file to extract the time and job ID for each emission. Assumes most recent emissions are at the end of the file.
    
    Args:
        emissions_file (str): Path to the CodeCarbon emissions log file.
        project_name (str): Name of the project to extract emissions for.
    
    Returns:
        pd.DataFrame: DataFrame containing the time and job ID for each emission.
    """
    df = pd.read_csv(emissions_file)
    df = df[df["project_name"] == project_name]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by='timestamp', ascending=False)

    if len(df) == 0:
        raise ValueError(f"No emissions found for project {project_name} in file {emissions_file}")

    duration = str(df['duration'].iloc[0])
    emissions = str(df['emissions'].iloc[0])

    return duration, emissions

def compute_balanced_atom_weights_from_frequencies(freq_dict: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Convert frequency dictionary to balanced weight dictionary with temperature scaling.
    
    Temperature closer to 0 makes weights more uniform, while higher temperatures 
    increase the relative weighting of rare tokens.
    
    Args:
        freq_dict: Dictionary mapping tokens to their frequencies
        temperature: Factor to scale frequency differences. Range (0, inf).
            temperature -> 0: weights become uniform
            temperature = 1: standard inverse frequency weights
            temperature > 1: amplifies differences between rare/common tokens
        
    Returns:
        Dictionary mapping tokens to weight values that average to 1.0
    """
    # Apply temperature scaling to frequencies
    scaled_freqs = {k: v**temperature for k,v in freq_dict.items()}
    
    # Convert frequencies to inverse weights
    weights = {k: 1/v if v > 0 else 1.0 for k,v in scaled_freqs.items()}
    
    # Normalize to mean 1.0
    mean_weight = sum(weights.values()) / len(weights)
    return {k: w/mean_weight for k,w in weights.items()}

def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print detailed tensor information for debugging.
    
    Args:
        name: Identifier for the tensor
        tensor: Tensor to analyze
    """
    print(f"\n=== {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Strides: {tensor.stride()}")
    print(f"Contiguous: {tensor.is_contiguous()}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    if tensor.grad is not None:
        print(f"Grad shape: {tensor.grad.shape}")
        print(f"Grad strides: {tensor.grad.stride()}")