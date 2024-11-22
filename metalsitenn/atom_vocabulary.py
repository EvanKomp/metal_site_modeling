# metalsitenn/atom_vocabulary.py
'''
* Author: Evan Komp
* Created: 11/4/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
"""metalsitenn/vocab.py"""

from typing import List, Union, Dict, Set
import torch
import numpy as np
from .constants import (METAL_IONS, RECORD_TYPES, ATOMS)

class BaseVocabulary:
    """Base vocabulary class for atomic properties."""
    def __init__(self, vocab: Set[str], include_mask: bool = True):
        self.stoi = {}
        if include_mask:
            self.stoi['<MASK>'] = 0
            self.mask_token = 0
        
        for token in sorted(vocab):
            self.stoi[token] = len(self.stoi)
            
        self.itos = {v:k for k,v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, items: Union[str, List[str]]) -> Union[int, torch.Tensor]:
        """Convert string(s) to token(s)."""
        try:
            if isinstance(items, str):
                return self.stoi[items]
            
            # vectorize over list
            # use multiple cpu if available
            return torch.tensor([self.stoi[item] for item in items], dtype=torch.long)
        except KeyError as e:
            raise KeyError(f"Token not found in vocabulary: {e}, vocabulary is {self.stoi}")

    def decode(self, tokens: Union[int, torch.Tensor]) -> Union[str, List[str]]:
        """Convert token(s) to string(s)."""
        if isinstance(tokens, (int, np.integer)):
            return self.itos.get(tokens, '<MASK>')
        return [self.itos.get(t.item(), '<MASK>') for t in tokens]

class AtomVocabulary(BaseVocabulary):
    """Maps atomic identities to integer tokens.
    
    Args:
        metal_known: If True, each metal gets unique token. If False, single 'METAL' token
        use_generic: If True, maps protein atoms to generic types (CA->C, OG->O, etc)
        include_hydrogen: If True, includes H in vocabulary. If False, H gets mask token
    
    Attributes:
        vocab_size: Size of vocabulary including special tokens 
        mask_token: Integer token for masking (always 0)
        metal_token: Integer token for generic metal if metal_known=False
    """
    def __init__(
        self, 
        metal_known: bool = True, 
        include_hydrogen: bool = True
    ): 
        
        # Filter hydrogen if not included
        vocab = ATOMS.copy()
        if not include_hydrogen:
            vocab.remove('H')
            
        super().__init__(vocab)
        
        # Add metals
        if metal_known:
            for metal in sorted(METAL_IONS):
                self.stoi[metal] = len(self.stoi)
            self.metal_token = None
        else:
            next_token = len(self.stoi)
            for metal in METAL_IONS:
                self.stoi[metal] = next_token
            self.metal_token = next_token
            
        self.itos = {v:k for k,v in self.stoi.items()}
        self.vocab_size = len(self.stoi)


class AtomTypeVocabulary(BaseVocabulary):
    """Vocabulary for atom record types (ATOM/HETATM)."""
    def __init__(self):
        super().__init__(RECORD_TYPES)


class AtomTokenizer:
    """Tokenizes atomic identities and record types."""
    def __init__(
            self,
            keep_hydrogen: bool = True,
            metal_known: bool = True
    ):
        self.atom_vocab = AtomVocabulary(metal_known, include_hydrogen=keep_hydrogen)
        self.record_vocab = AtomTypeVocabulary()

        self.mask_token = self.atom_vocab.mask_token

        self.oh_size = self.atom_vocab.vocab_size + self.record_vocab.vocab_size


    def tokenize(self, atoms: List[List[str]], atom_types: List[List[str]]) -> Dict[str, torch.Tensor]:
        """Converts atom names and record types to integer tokens.
        
        Args:
            atoms: List of [num_atoms] atom names
            atom_types: List of [num_atoms] record types

        Returns:
            Dictionary with keys 'atoms' and 'records' containing tokenized data
        """
        return {
            'atoms': [self.atom_vocab.encode(names) for names in atoms],
            'atom_types': [self.record_vocab.encode(types) for types in atom_types]
        }
    
    def decode(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Converts integer tokens back to atom names and record types.
        
        Args:
            tokens: Dictionary with keys 'atoms' and 'records' containing tokenized data
        
        Returns:
            Dictionary with keys 'atoms' and 'records' containing decoded data
        """
        return {
            'atoms': [self.atom_vocab.decode(t) for t in tokens['atoms']],
            'atom_types': [self.record_vocab.decode(t) for t in tokens['atom_types']]
        }