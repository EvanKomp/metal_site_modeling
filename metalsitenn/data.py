# metalsitenn/data.py
'''
* Author: Evan Komp
* Created: 11/6/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
from biopandas.pdb import PandasPdb
import numpy as np
from typing import Tuple, List, Dict, Optional
import re
import pandas as pd

import torch

import logging
logger = logging.getLogger(__name__)



class PDBReader:
    """Reads PDB files to extract atomic coordinates and metadata.
    
    Uses BioPandas to efficiently parse PDB files and extract:
    - 3D positions
    - Full atom names (e.g. 'CA', 'CB')
    - Base elements (e.g. 'C', 'N') 
    - Record types (ATOM/HETATM)
    """
    
    def __init__(self, deprotonate: bool = False, skip_only_hetatm: bool = True):
        self.parser = PandasPdb()
        self.deprotonate = deprotonate
        self.skip_only_hetatm = skip_only_hetatm
        
    def read(self, pdb_path: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """Read PDB file and extract atomic information.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            positions: [N,3] array of atomic coordinates
            atom_names: List of full atom names (e.g. 'CA', 'CB')
            elements: List of base elements (e.g. 'C', 'N')
            record_types: List of ATOM/HETATM record types
        """
        structure = self.parser.read_pdb(pdb_path)
        df = pd.concat([structure.df['ATOM'], structure.df['HETATM']])
        if self.deprotonate:
            df = df[~(df['element_symbol'].isin(['H', 'D']))]
        
        positions = df[['x_coord', 'y_coord', 'z_coord']].values
        atom_names = df['atom_name'].tolist()
        record_types = df['record_name'].tolist()
        symbols = df['element_symbol'].tolist()
        
        return {
            'positions': positions,
            'atom_names': atom_names,
            'atoms': symbols,
            'atom_types': record_types
        }
    
    def read_dir(self, pdb_dir: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """Read all PDB files in a directory as an iterator"""

        for file in os.listdir(pdb_dir):
            if file.endswith(".pdb"):
                outs = self.read(os.path.join(pdb_dir, file))
                outs['id'] = file.split('.')[0]
                if self.skip_only_hetatm and all([x == 'HETATM' for x in outs['atom_types']]):
                    continue
                yield outs


class AtomicSystemBatchCollator:
    """Processes batches of atomic systems with optional masking and noise.

    Handles HuggingFace dataset batches (dict of lists) and converts to dict of tensors.
    Optionally applies atom masking and position noise during training.

    Args:
        tokenizer: Tokenizer instance providing mask token and vocabulary
        mask_rate: Fraction of atoms to mask during training, if None no masking
        noise_rate: Fraction of positions to add noise, if None no noise
        zero_noise_in_loss_rate: Additional positions to include in loss but not noise 
        noise_scale: Standard deviation of gaussian noise (Angstroms)
        already_tokenized: If True, input is already tokenized, otherwise tokenizes
        return_original_positions: If True, returns original positions before noise

    Input batch format:
        {
            'atoms': List[List[int]], # Atomic number tokens
            'atom_types': List[List[int]], # ATOM/HETATM tokens  
            'positions': List[List[float]], # xyz coordinates
            'id': List[str] # Optional system identifiers
        }

    Output format:
        {
            'atoms': [n_atoms_total] atom tokens, possibly masked
            'atom_types': [n_atoms_total] record tokens
            'positions': [n_atoms_total, 3] coordinates
            'batch_indices': [n_atoms_total] batch indices
            'mask_mask': [n_atoms] mask of which atoms are masked
            'atom_labels': [n_atoms] labels for masked atoms
            'atom_type_labels': [n_atoms] labels for masked atom types
            'noise_mask': [n_atoms] mask of which atoms are noised, if noising
            'denoise_vectors': [n_atoms, 3] vectors required to denoise positions, if noising, for computing loss, this vector is scaled.
            'noise_loss_mask': [n_atoms] mask of which atoms are used for loss, if noising
            'id': [batch_size] original system IDs if provided
            any other fields in input batch
        }
    """

    def __init__(
            self,
            tokenizer,
            mask_rate: Optional[float] = None,
            noise_rate: Optional[float] = None, 
            zero_noise_in_loss_rate: Optional[float] = None,
            noise_scale: float = 1.0,
            already_tokenized: bool = True,
            return_original_positions: bool = False
        ):
        self.atom_mask_token = tokenizer.atom_mask_token
        self.type_mask_token = tokenizer.type_mask_token
        self.mask_rate = mask_rate
        self.noise_rate = noise_rate
        self.zero_noise_in_loss_rate = zero_noise_in_loss_rate
        self.noise_scale = noise_scale
        self.already_tokenized = already_tokenized
        self.return_original_positions = return_original_positions


    def __call__(self, batch: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Process a batch of atomic systems.

        Args:
            batch: HuggingFace format batch dictionary

        Returns:
            Dictionary of processed tensors with optional masking/noise
        """
        # Track batch sizes for creating index tensor
        batch_sizes = [len(atoms) for atoms in batch['atoms']]
        total_atoms = sum(batch_sizes)
        logger.debug(f"Processing batch with {len(batch_sizes)} systems, {total_atoms} total atoms")

        # Create batch index tensor
        batch_idx = torch.repeat_interleave(
            torch.arange(len(batch_sizes)), 
            torch.tensor(batch_sizes)
        )

        # tokenize if necessary
        if not self.already_tokenized:
            batch['atoms'] = [self.tokenizer.encode(x) for x in batch['atoms']]
            batch['atom_types'] = [self.tokenizer.encode(x) for x in batch['atom_types']]

        # Concatenate and convert to tensors
        output = {
            'atoms': torch.cat([torch.tensor(x) for x in batch['atoms']]),
            'atom_types': torch.cat([torch.tensor(x) for x in batch['atom_types']]),
            'positions': torch.cat([torch.tensor(x) for x in batch['positions']]),
            'batch_indices': batch_idx
        }

        # Apply masking
        if self.mask_rate and self.mask_rate > 0:
            output['atom_labels'] = output['atoms'].clone()
            output['atom_type_labels'] = output['atom_types'].clone()

            n_mask = int(total_atoms * self.mask_rate)
            mask_idx = torch.randperm(total_atoms)[:n_mask]
            
            output['atoms'][mask_idx] = self.atom_mask_token
            output['atom_types'][mask_idx] = self.type_mask_token
            output['mask_mask'] = torch.zeros(total_atoms, dtype=torch.bool)
            output['mask_mask'][mask_idx] = True
            
            logger.debug(f"Masked {n_mask} atoms")

        # Apply coordinate noise
        if self.noise_rate and self.noise_rate > 0:
            n_noise = int(total_atoms * self.noise_rate)
            randperm = torch.randperm(total_atoms)
            noise_idx = randperm[:n_noise]
            
            # Additional positions for loss but no noise
            if self.zero_noise_in_loss_rate:
                n_zero_noise = int(total_atoms * self.zero_noise_in_loss_rate)
                zero_noise_idx = randperm[n_noise:n_noise+n_zero_noise]
                noise_loss_idx = torch.cat([noise_idx, zero_noise_idx])
            else:
                noise_loss_idx = noise_idx


            if self.return_original_positions:
                output['original_positions'] = output['positions'].clone()

            noise_vectors = torch.zeros_like(output['positions'])
            noise_vectors[noise_idx] = torch.randn(n_noise, 3)
            # move the atoms by noise times scale
            # return vectors will not be scaled so model can be trained with low activations
            output['positions'] = output['positions'] + noise_vectors * self.noise_scale
            denoise_loss_vectors = noise_vectors * -1
            output['denoise_vectors'] = denoise_loss_vectors
            output['noise_loss_mask'] = torch.zeros(total_atoms, dtype=torch.bool)
            output['noise_loss_mask'][noise_loss_idx] = True
            output['noise_mask'] = torch.zeros(total_atoms, dtype=torch.bool)
            output['noise_mask'][noise_idx] = True

            logger.debug(f"Added noise to {n_noise} positions, tracking loss on {len(noise_loss_idx)} positions")
        
        # pass through other keys
        for key, value in batch.items():
            if key not in output:
                output[key] = value

        return output