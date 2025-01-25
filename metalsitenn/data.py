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
from typing import Tuple, List, Dict, Optional, Any, Union
import re
import pandas as pd
from collections import defaultdict

import torch

from metalsitenn.atom_vocabulary import AtomTokenizer

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
            pos: [N,3] array of atomic coordinates
            atom_names: List of full atom names (e.g. 'CA', 'CB')
            atoms: List of base elements (e.g. 'C', 'N')
            atom_types: List of ATOM/HETATM record types
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
            'pos': positions,
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
    
    Supports two input formats:
    1. Dictionary of lists:
        {
            'atoms': List[List[int]], # Atomic number tokens  
            'atom_types': List[List[int]], # ATOM/HETATM tokens
            'pos': List[List[float]], # [N,3] xyz coordinates
            'id': List[str] # Optional system identifiers
        }
        
    2. List of dictionaries:
        [
            {
                'atoms': List[int],
                'atom_types': List[int],
                'pos': List[float],
                'id': str
            },
            ...
        ]
        
    Output format:
        {
            'atoms': torch.Tensor, # [N] atom tokens, possibly masked
            'atom_types': torch.Tensor, # [N] record tokens  
            'pos': torch.Tensor, # [N,3] coordinates
            'batch_idx': torch.Tensor, # [N] batch indices
            'mask': torch.Tensor, # [N] boolean mask of masked atoms
            'atom_labels': torch.Tensor, # [N] labels for masked atoms
            'atom_type_labels': torch.Tensor, # [N] labels for masked types
            'noise_mask': torch.Tensor, # [N] boolean mask of noised atoms
            'denoise_vectors': torch.Tensor, # [N,3] vectors to denoise positions  
        }
    """

    def __init__(
        self,
        tokenizer: AtomTokenizer,
        mask_rate: Optional[float] = None,
        noise_rate: Optional[float] = None,
        noise_scale: float = 1.0,
        already_tokenized: bool = True,
        return_original_pos: bool = False
    ):
        self.atom_mask_token = tokenizer.atom_mask_token
        self.type_mask_token = tokenizer.type_mask_token
        self.mask_rate = mask_rate
        self.noise_rate = noise_rate
        self.noise_scale = noise_scale
        self.already_tokenized = already_tokenized
        self.return_original_pos = return_original_pos
        self.tokenizer = tokenizer

    def _standardize_batch(self, batch: Union[Dict[str, List[Any]], List[Dict[str, Any]]]) -> Dict[str, List[Any]]:
        """Convert both input formats to dictionary of lists format."""
        if isinstance(batch, dict):
            return batch
        
        # Convert list of dicts to dict of lists
        output = defaultdict(list)
        for item in batch:
            for key, value in item.items():
                output[key].append(value)
        return dict(output)

    def __call__(self, batch: Union[Dict[str, List[Any]], List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Process a batch of atomic systems."""
        # Standardize input format
        batch = self._standardize_batch(batch)
        
        batch_sizes = [len(atoms) for atoms in batch['atoms']]
        num_atoms = sum(batch_sizes)

        batch_idx = torch.repeat_interleave(
            torch.arange(len(batch_sizes)), 
            torch.tensor(batch_sizes)
        )

        if not self.already_tokenized:
            batch['atoms'] = [self.tokenizer.encode(x) for x in batch['atoms']]
            batch['atom_types'] = [self.tokenizer.encode(x) for x in batch['atom_types']]

        output = {
            'atoms': torch.cat([torch.tensor(x) for x in batch['atoms']]),
            'atom_types': torch.cat([torch.tensor(x) for x in batch['atom_types']]),
            'pos': torch.cat([torch.tensor(x) for x in batch['pos']]),
            'batch_idx': batch_idx
        }

        if self.mask_rate and self.mask_rate > 0:
            output['atom_labels'] = output['atoms'].clone()
            output['atom_type_labels'] = output['atom_types'].clone()

            n_mask = int(num_atoms * self.mask_rate)
            mask_idx = torch.randperm(num_atoms)[:n_mask]
            
            output['atoms'][mask_idx] = self.atom_mask_token
            output['atom_types'][mask_idx] = self.type_mask_token
            output['mask'] = torch.zeros(num_atoms, dtype=torch.bool)
            output['mask'][mask_idx] = True

        if self.noise_rate and self.noise_rate > 0:
            n_noise = int(num_atoms * self.noise_rate)
            randperm = torch.randperm(num_atoms)
            noise_idx = randperm[:n_noise]

            if self.return_original_pos:
                output['original_pos'] = output['pos'].clone()

            noise_vectors = torch.zeros_like(output['pos'])
            noise_vectors[noise_idx] = torch.randn(n_noise, 3) * self.noise_scale
            output['pos'] = output['pos'] + noise_vectors
            output['denoise_vectors'] = noise_vectors * -1
            output['noise_mask'] = torch.zeros(num_atoms, dtype=torch.bool)
            output['noise_mask'][noise_idx] = True

        for key, value in batch.items():
            if key not in output:
                output[key] = value

        return output