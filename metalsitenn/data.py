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
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RadiusGraph

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
