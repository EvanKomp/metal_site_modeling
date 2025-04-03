# pipeline/1.4_compute_null_model.py
'''
* Author: Evan Komp
* Created: 1/30/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology

Computes baseline performance metrics using most frequent token prediction and zero vector noise prediction.

Outputs:
    data/metrics/null_model_metrics.json:
        - mask_accuracy: Accuracy of most common token prediction
        - type_accuracy: Accuracy of most common record type prediction  
        - noise_mae: mae of zero vector noise prediction
'''
import json
from datasets import load_from_disk
from collections import Counter
from tqdm import tqdm
import numpy as np
import dvc.api
import joblib
import torch

from metalsitenn.data import AtomicSystemBatchCollator
from metalsitenn.utils import ParamsObj

def count_masked_tokens(dataset, collator, key):
    """Count token frequencies of masked positions."""
    counter = Counter()
    
    for example in tqdm(dataset, desc=f"Counting {key}"):
        # Apply masking using collator
        batch = collator([example])
        mask = batch['mask']
        labels = batch[key]
        
        counter.update(labels[mask].numpy())
            
    return counter, counter.most_common(1)[0][0]

def compute_metrics(dataset, collator, most_common_atom, most_common_type):
    """Compute metrics using consistent masking."""
    correct_atoms = 0
    correct_types = 0
    total_masks = 0
    total_mae = 0
    total_systems = 0
    
    for example in tqdm(dataset, desc="Computing metrics"):
        batch = collator([example])
        mask = batch['mask']
        
        if not any(mask):
            continue

        # on a per atom basis because thats what the eval metric mask accuracy is    
        mask_indices = np.where(mask)[0]
        total_masks += len(mask_indices)
        
        correct_atoms += torch.sum(
            batch['atom_labels'][mask_indices] == most_common_atom
        ).item()
        correct_types += torch.sum(
            batch['atom_type_labels'][mask_indices] == most_common_type
        ).item()
        
        # on a per system basis because thats what the eval metric noise loss is
        if any(batch['noise_mask']):
            total_mae += torch.mean(
                torch.sqrt((batch['denoise_vectors'][batch['noise_mask']]**2).sum(dim=1))
            ).item()
            total_systems += 1

    return {
        'mask_accuracy': correct_atoms / total_masks,
        'type_accuracy': correct_types / total_masks,
        'noise_mae': total_mae / total_systems if total_systems > 0 else 0
    }

def main():
    # Load params and setup collator
    params = ParamsObj(dvc.api.params_show())
    tokenizer = joblib.load("data/dataset/tokenizer.pkl")
    
    collator = AtomicSystemBatchCollator(
        tokenizer=tokenizer,
        mask_rate=params.training.mask_rate,
        noise_rate=params.training.noise_rate,
        noise_scale=params.training.noise_scale,
        already_tokenized=True
    )
    
    # Load dataset
    if not params.training.debug_use_toy:
        dataset = load_from_disk("data/dataset/metal_site_dataset")
    else:
        dataset = load_from_disk("data/toy_dataset")
    
    # Get most common tokens from training set
    atom_counter, most_common_atom = count_masked_tokens(dataset["train"], collator, 'atom_labels')
    _, most_common_type = count_masked_tokens(dataset["train"], collator, 'atom_type_labels')
    
    # Compute metrics on test set  
    metrics = compute_metrics(dataset["test"], collator, most_common_atom, most_common_type)

    # also add fractions of each element
    atom_fractions = {k: v / sum(atom_counter.values()) for k, v in atom_counter.items()}
    # convert using tokenizer
    atom_fractions = {tokenizer.atom_vocab.itos[k]: v for k, v in atom_fractions.items()}
    metrics.update(atom_fractions)
    
    with open("data/metrics/null_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()