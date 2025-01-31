# metalsitenn/io.py
'''
* Author: Evan Komp
* Created: 1/30/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import Dict, Tuple, List
import torch
from biopandas.pdb import PandasPdb
import numpy as np
import pandas as pd

from metalsitenn.data import PDBReader 
from metalsitenn.atom_vocabulary import AtomTokenizer

def load_metal_site(pdb_path: str, tokenizer: AtomTokenizer, deprotonate: bool = True) -> Dict[str, torch.Tensor]:
    """Load single metal binding site into model-ready format.
    
    Args:
        pdb_path: Path to PDB file
        tokenizer: Initialized AtomTokenizer 
        deprotonate: Remove hydrogens
        
    Returns:
        Dict containing:
            atoms: [N] atom type tokens
            atom_types: [N] record type tokens
            pos: [N,3] coordinates
            residue_info: List[Dict] containing per-atom information:
                - residue_number: int
                - residue_name: str
                - chain_id: str
                - atom_name: str
                - insertion: str
                - alt_loc: str
                - occupancy: float 
                - b_factor: float
                - segment_id: str
    """
    # Read PDB
    pdb = PandasPdb().read_pdb(pdb_path)
    df = pd.concat([pdb.df['ATOM'], pdb.df['HETATM']])
    if deprotonate:
        df = df[~df['element_symbol'].isin(['H', 'D'])]
        
    residue_info = [
        {
            'residue_number': int(row['residue_number']),
            'residue_name': str(row['residue_name']),
            'chain_id': str(row['chain_id']),
            'atom_name': str(row['atom_name']),
            'insertion': str(row['insertion']) if pd.notna(row['insertion']) else '',
            'alt_loc': str(row['alt_loc']) if pd.notna(row['alt_loc']) else '',
            'occupancy': float(row['occupancy']) if pd.notna(row['occupancy']) else 1.0,
            'b_factor': float(row['b_factor']) if pd.notna(row['b_factor']) else 0.0,
            'segment_id': str(row['segment_id']) if pd.notna(row['segment_id']) else ''
        }
        for _, row in df.iterrows()
    ]
    
    # Get atomic data
    site_data = {
        'atoms': df['element_symbol'].tolist(),
        'atom_types': df['record_name'].tolist(),
        'pos': df[['x_coord', 'y_coord', 'z_coord']].values,
    }
    
    # Tokenize
    tokens = tokenizer.tokenize(
        atoms=[site_data['atoms']], 
        atom_types=[site_data['atom_types']]
    )
    
    return {
        'atoms': tokens['atoms'][0],
        'atom_types': tokens['atom_types'][0], 
        'pos': torch.tensor(site_data['pos']),
        'residue_info': residue_info
    }

def save_metal_site(atoms: torch.Tensor, atom_types: torch.Tensor, pos: torch.Tensor,
                   residue_info: List[Dict], tokenizer: AtomTokenizer, output_path: str,
                   append: bool = False, model_num: int = 1) -> None:
    """Save model outputs as PDB file or append as new frame.
    
    Args:
        atoms: Atom type tokens
        atom_types: Record type tokens
        pos: Atomic coordinates 
        residue_info: Per-atom PDB metadata
        tokenizer: Tokenizer for decoding
        output_path: Path to output PDB
        append: If True, append as new MODEL, else create new file
        model_num: Model number when appending
    """
    decoded = tokenizer.decode(atoms=atoms, atom_types=atom_types)
    records = []
    
    if append:
        mode = 'a'
        records.append(f"MODEL     {model_num}\n")
    else:
        mode = 'w'
        
    for i in range(len(atoms)):
        atom_symbol = decoded['atoms'][i]
        if atom_symbol == '<METAL>':
            atom_symbol = 'CU'
            residue_name = 'CU'
        else:
            residue_name = residue_info[i]['residue_name']
            
        line = (f"{decoded['atom_types'][i]:<6}"
                f"{i+1:>5} "
                f"{residue_info[i]['atom_name']:<4}"
                f"{'':<1}"
                f"{residue_name:>3} "
                f"{residue_info[i]['chain_id']:>1}"
                f"{residue_info[i]['residue_number']:>4}"
                f"{residue_info[i].get('insertion',''):>1}   "
                f"{float(pos[i,0].item()):>8.3f}"
                f"{float(pos[i,1].item()):>8.3f}"
                f"{float(pos[i,2].item()):>8.3f}"
                f"{float(residue_info[i].get('occupancy',1.0)):>6.2f}"
                f"{float(residue_info[i].get('b_factor',0.0)):>6.2f}"
                f"{'':>11}"
                f"{atom_symbol:>2}\n")
        records.append(line)
        
    if append:
        records.append("ENDMDL\n")
        
    with open(output_path, mode) as f:
        f.writelines(records)