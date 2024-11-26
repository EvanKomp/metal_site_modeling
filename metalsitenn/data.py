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
from typing import Tuple, List
import re
import pandas as pd

import torch
from torch_geometric.data import Data, Batch

class PDBReader:
    """Reads PDB files to extract atomic coordinates and metadata.
    
    Uses BioPandas to efficiently parse PDB files and extract:
    - 3D positions
    - Full atom names (e.g. 'CA', 'CB')
    - Base elements (e.g. 'C', 'N') 
    - Record types (ATOM/HETATM)
    """
    
    def __init__(self, deprotonate: bool = False):
        self.parser = PandasPdb()
        self.deprotonate = deprotonate  

        
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
            df = df[~(df['element_symbol'] == 'H')]
        
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
                yield outs


class AtomicSystemCollator:
    """Collates atomic system data with optional masking and position noising.
    
    Args:
        tokenizer: AtomTokenizer instance for mask token
        max_radius: Maximum radius for graph construction (Å)
        mask_rate: Fraction of atoms to mask (default: None)
        noise_rate: Fraction of positions to add noise to (default: None)  
        noise_width: Standard deviation of gaussian noise (default: 0.1Å)
    """
    def __init__(
        self,
        tokenizer,
        mask_rate: float = None,
        noise_rate: float = None,
        noise_width: float = 0.1
    ):
        self.mask_token = tokenizer.mask_token
        self.mask_rate = mask_rate
        self.noise_rate = noise_rate
        self.noise_width = noise_width

    def __call__(self, batch: List[dict]) -> Batch:
        """Collate batch of atomic systems with masking and noise.
        
        Args:
            batch: List of dicts with keys:
                - atoms: Tensor of atom type indices 
                - atom_types: Tensor of ATOM/HETATM indices
                - positions: Tensor of 3D coordinates
                - label_*: Any label tensors
        
        Returns:
            PyG Batch object with:
                - atoms: Atom type indices (masked if mask_rate > 0)  
                - atom_types: ATOM/HETATM record indices
                - pos: Noised positions if noise_rate > 0
                - original_pos: Original positions if noise applied
                - edge_index: [2, num_edges] COO format edge indices
                - batch: Batch assignment for each node
                - mask_mask: Mask for masked atoms
                - noise_mask: Mask for noised positions
                - label_*: Original label tensors
        """
        # Create list of Data objects
        data_list = []
        for i, d in enumerate(batch):
            data = Data(
                atoms=d['atoms'],
                atom_types=d['atom_types'], 
                pos=d['positions']
            )
            
            # Add any labels
            for k, v in d.items():
                if k.startswith('label_'):
                    data[k] = v
                    
            data_list.append(data)

        # Batch the data
        batch = Batch.from_data_list(data_list)
        
        # Apply masking
        if self.mask_rate:
            n_mask = int(len(batch.atoms) * self.mask_rate)
            mask_idx = torch.randperm(len(batch.atoms))[:n_mask]
            batch.atoms[mask_idx] = self.mask_token
            batch.atom_types[mask_idx] = self.mask_token
            #conver indexes into a bool mask
            batch.atom_mask = torch.zeros(len(batch.atoms), dtype=torch.bool)
            batch.atom_mask[mask_idx] = True

        else:
            batch.atom_mask = None

        # Apply position noise  
        if self.noise_rate:
            n_noise = int(len(batch.pos) * self.noise_rate)
            noise_idx = torch.randperm(len(batch.pos))[:n_noise]
            batch.original_pos = batch.pos[noise_idx].clone()
            batch.pos[noise_idx] += torch.randn_like(batch.pos[noise_idx]) * self.noise_width
            batch.noise_mask = torch.zeros(len(batch.pos), dtype=torch.bool)
            batch.noise_mask[noise_idx] = True
        else:
            batch.noise_mask = None

        batch.atom_tokens = torch.vstack([batch.atoms, batch.atom_types]).T

        return batch