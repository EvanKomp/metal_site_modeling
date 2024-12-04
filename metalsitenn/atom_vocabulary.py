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
from .constants import (METAL_IONS, RECORD_TYPES, COMMON_PROTEIN_ATOMS, UNCOMMON_PROTEIN_ATOMS)

import logging
logger = logging.getLogger(__name__)

# unknonw token error
class UnknownTokenError(Exception):
    pass

class BaseVocabulary:
    """Base vocabulary class for atomic properties."""
    def __init__(self, vocab: Set[str], include_mask: bool = True, allow_unknown: bool = False):
        self.stoi = {}
        if include_mask:
            self.stoi['<MASK>'] = 0
            self.mask_token = 0

        self.allow_unknown = allow_unknown
        if allow_unknown:
            next_token = self._get_next_token_index()
            self.stoi['<UNK>'] = next_token
            self.stoi['X'] = next_token
            self.unk_token = next_token
        
        for token in sorted(vocab):
            self.stoi[token] = self._get_next_token_index()
            
        self.itos = {v:k for k,v in self.stoi.items()}
        self.vocab_size = self._get_next_token_index()

    def _get_next_token_index(self):
        """Multiple strings may map to the same token, so we need to find the next available token index."""
        unique_tokens = set(self.stoi.values())
        return max(unique_tokens) + 1

    def encode(self, items: Union[str, List[str]]) -> Union[int, torch.Tensor]:
        """Convert string(s) to token(s)."""
        try:
            if isinstance(items, str):
                return self.stoi[items]
            
            # vectorize over list
            # use multiple cpu if available
            return torch.tensor([self.stoi[item] for item in items], dtype=torch.long)
        except KeyError as e:
            if self.allow_unknown:
                logger.warning(f"Unknown token: {e}")
                return self.unk_token
            raise UnknownTokenError(f"Unknown token: {e}, available are {self.stoi.keys()}")

    def decode(self, tokens: Union[int, torch.Tensor]) -> Union[str, List[str]]:
        """Convert token(s) to string(s)."""
        if isinstance(tokens, (int, np.integer)):
            return self.itos.get(tokens, '<MASK>')
        return [self.itos.get(t.item(), '<MASK>') for t in tokens]

class AtomVocabulary(BaseVocabulary):
    """Maps atomic identities to integer tokens.
    
    Args:
        metal_known: If True, each metal gets unique token. If False, single 'METAL' token
        include_hydrogen: If True, include hydrogen atoms in vocabulary
        aggregate_uncommon: If True, aggregate uncommon elements into a single token
        allow_unknown: If True, allow unknown tokens to be mapped to <UNK>, otherwise throw
    
    Attributes:
        vocab_size: Size of vocabulary including special tokens 
        mask_token: Integer token for masking (always 0)
        metal_token: Integer token for generic metal if metal_known=False
    """

    def __init__(
        self, 
        metal_known: bool = True, 
        include_hydrogen: bool = True,
        aggregate_uncommon: bool = False,
        allow_unknown: bool = False
    ): 
        
        # Filter hydrogen if not included
        vocab = COMMON_PROTEIN_ATOMS
        if not include_hydrogen:
            vocab.remove('H')
            vocab.remove('D')
            
        super().__init__(vocab, include_mask=True, allow_unknown=allow_unknown)

        # Add metals
        if metal_known:
            for metal in sorted(METAL_IONS):
                self.stoi[metal] = self._get_next_token_index()
            self.metal_token = None
        else:
            next_token = self._get_next_token_index()
            for metal in METAL_IONS:
                self.stoi[metal] = next_token
            self.metal_token = next_token

        # add uncommon elements
        if not aggregate_uncommon:
            for atom in sorted(UNCOMMON_PROTEIN_ATOMS):
                self.stoi[atom] = self._get_next_token_index()
            self.uncommon_token = None
        else:
            next_token = self._get_next_token_index()
            for atom in UNCOMMON_PROTEIN_ATOMS:
                self.stoi[atom] = next_token
            self.uncommon_token = next_token
            
        self.itos = {v:k for k,v in self.stoi.items()}

        # update the metal token and uncommon token
        # to reflect if they were combined, otherwise the last added 
        # metal or uncommon would be in itos
        if not metal_known:
            self.itos[self.metal_token] = '<METAL>'
        if aggregate_uncommon:
            self.itos[self.uncommon_token] = '<UNCOMMON>'

        # get vocab size
        self.vocab_size = self._get_next_token_index()


class AtomTypeVocabulary(BaseVocabulary):
    """Vocabulary for atom record types (ATOM/HETATM)."""
    def __init__(self):
        super().__init__(RECORD_TYPES)


class AtomTokenizer:
    """Tokenizes atomic identities and record types."""
    def __init__(
            self,
            keep_hydrogen: bool = True,
            metal_known: bool = True,
            aggregate_uncommon: bool = False,
            allow_unknown: bool = False
    ):
        self.atom_vocab = AtomVocabulary(
            metal_known=metal_known,
            include_hydrogen=keep_hydrogen,
            aggregate_uncommon=aggregate_uncommon,
            allow_unknown=allow_unknown
        )
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