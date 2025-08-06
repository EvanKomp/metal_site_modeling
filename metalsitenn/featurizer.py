# metalsitenn/featurizer.py
'''
* Author: Evan Komp
* Created: 6/17/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import List, Dict, Tuple, Any
import torch
import networkx as nx
import warnings

from metalsitenn.tokenizers import TOKENIZERS, Tokenizer
from metalsitenn.placer_modules.cifutils import Chain, mutate_chain
from metalsitenn.constants import I2E

from metalsitenn.graph_data import ProteinData, make_top_k_graph

class MetalSiteFeaturizer:
    """
    Featurizer for metal sites - converts Chain objects into tokenized graphs and assigns them to ProteinData objects.
    

    Additional functionalities:
    - masking atom identities
      - also returns labels and loss mask
    - collapse atoms from residues and noise the
      - also returns mask of which atoms were collapsed / moved
    - Mutate particular residues by replacing them with a different residue type and noising them


    Params
    - atom_features: List[str] - list of atom features to include in the featurization
    - bond_features: List[str] - list of bond features to include in the featurization
    - k: int - Number of atoms in each graph
    """
    tokenizers = TOKENIZERS

    def __init__(
            self, 
            atom_features: List[str] = ['element'],
            bond_features: List[str] = ['bond_order'],
            k: int = 20
    ):
        # Validate all requested features exist
        all_features = atom_features + bond_features
        missing_features = [f for f in all_features if f not in self.tokenizers]
        if missing_features:
            raise ValueError(f"Unknown features: {missing_features}. "
                           f"Available features: {list(self.tokenizers.keys())}")
        
        self.atom_features = atom_features.copy()
        self.bond_features = bond_features.copy()
        
        # Validate that bond features have non_bonded_token_id
        for feature in self.bond_features:
            tokenizer = self.tokenizers[feature]
            if not hasattr(tokenizer, 'non_bonded_token_id'):
                raise ValueError(f"Bond feature '{feature}' tokenizer must have non_bonded_token_id attribute")
            
        self.k = k

    
