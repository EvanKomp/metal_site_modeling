# metalsitenn/atom_vocabulary.py
'''
* Author: Evan Komp
* Created: 11/4/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
"""metalsitenn/vocab.py"""

from typing import List, Union, Dict, Set, Optional
import torch
import numpy as np
from .constants import (METAL_IONS, RECORD_TYPES, COMMON_PROTEIN_ATOMS, UNCOMMON_PROTEIN_ATOMS)
from .utils import compute_balanced_atom_weights_from_frequencies
import copy
from pathlib import Path
import json

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

        if isinstance(items, str):
            return self.stoi[items]
        
        toks = []
        for item in items:
            try:
                toks.append(self.stoi[item])
            except KeyError:
                if self.allow_unknown:
                    toks.append(self.stoi['<UNK>'])
                else:
                    raise UnknownTokenError(f"Unknown token: {item}, available are {self.stoi.keys()}")
        
        return torch.tensor(toks, dtype=torch.long)

    def decode(self, tokens: Union[int, torch.Tensor]) -> Union[str, List[str]]:
        """Convert token(s) to string(s)."""
        if isinstance(tokens, (int, np.integer)):
            return self.itos.get(tokens, None)
        elif isinstance(tokens, torch.Tensor):
            return [self.itos.get(t.item(), None) for t in tokens]
        elif hasattr(tokens, '__iter__'):
            # asuming an iter of tokens as opposed to an iter of iter
            if isinstance(tokens[0], (int, np.integer)):
                return [self.itos.get(t, None) for t in tokens]
            elif hasattr(tokens[0], '__iter__'):
                return [self.decode(t) for t in tokens]

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
        vocab = copy.copy(COMMON_PROTEIN_ATOMS)
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
            self.stoi['<METAL>'] = next_token
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
            allow_unknown: bool = False,
            name_or_path: Optional[str] = None
    ):
        self.atom_vocab = AtomVocabulary(
            metal_known=metal_known,
            include_hydrogen=keep_hydrogen,
            aggregate_uncommon=aggregate_uncommon,
            allow_unknown=allow_unknown
        )
        self.record_vocab = AtomTypeVocabulary()

        self.atom_mask_token = self.atom_vocab.mask_token
        self.type_mask_token = self.record_vocab.mask_token

        self.oh_size = self.atom_vocab.vocab_size + self.record_vocab.vocab_size

        self.init_kwargs = {
            "keep_hydrogen": keep_hydrogen,
            "metal_known": metal_known, 
            "aggregate_uncommon": aggregate_uncommon,
            "allow_unknown": allow_unknown
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from saved files.
        
        Args:
            pretrained_model_name_or_path: Directory containing vocab files
            **kwargs: Override saved config parameters
        """
        vocab_files = cls.get_vocab_files(pretrained_model_name_or_path)
        
        # Load config
        with open(vocab_files["config_file"], "r") as f:
            config = json.load(f)
            
        # Override with kwargs
        config.update(kwargs)
        
        return cls(**config)

    def save_pretrained(self, save_directory: str):
        """Save tokenizer vocabulary and configuration.
        
        Args:
            save_directory: Directory to save files
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(exist_ok=True)

        # Save config
        with open(save_dir / "tokenizer_config.json", "w") as f:
            json.dump(self.init_kwargs, f, indent=2)

    def get_vocab(self) -> Dict[str, int]:
        """Get combined vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to ids
        """
        vocab = {}
        vocab.update(self.atom_vocab.stoi)
        vocab.update(self.record_vocab.stoi)
        return vocab

    @staticmethod
    def get_vocab_files(save_directory: str) -> Dict[str, str]:
        """Get vocab file paths.
        
        Args:
            save_directory: Directory containing vocab files
            
        Returns:
            Dict mapping file types to paths
        """
        save_dir = Path(save_directory)
        return {
            "config_file": str(save_dir / "tokenizer_config.json"),
        }

    def tokenize(self, atoms: List[List[str]], atom_types: List[List[str]]) -> Dict[str, torch.Tensor]:
        """Converts atom names and record types to integer tokens.
        
        Args:
            atoms: List of [num_atoms] atom names
            atom_types: List of [num_atoms] record types

        Returns:
            Dictionary with keys 'atoms' and 'records' containing tokenized data
        """
        outs = {
            'atoms': [self.atom_vocab.encode(names) for names in atoms],
            'atom_types': [self.record_vocab.encode(types) for types in atom_types]
        }
        # check that lengths match for each
        for i in range(len(outs['atoms'])):
            if len(outs['atoms'][i]) != len(outs['atom_types'][i]):
                raise ValueError(f"Tokenization created inconsistent lengths: atoms={len(outs['atoms'][i])}, atom_types={len(outs['atom_types'][i])}")
        
        return outs
    
    def _decode_atoms(self, atoms: List[int]) -> List[str]:
        """Converts integer tokens back to atom names."""
        return self.atom_vocab.decode(atoms)
    
    def _decode_records(self, records: List[int]) -> List[str]:
        """Converts integer tokens back to record types."""
        return self.record_vocab.decode(records)

    def decode(self, atoms: Union[int, torch.Tensor]=None, atom_types: Union[int, torch.Tensor]=None) -> Dict[str, List[str]]:
        """Converts integer tokens back to atom names and record types.
        
        Args:
            atoms: Integer tensor of atom tokens
            atom_types: Integer tensor of record type tokens

        Returns:
            Dictionary with keys 'atoms' and 'records' containing decoded data
        """
        return {
            'atoms': self._decode_atoms(atoms) if atoms is not None else None,
            'atom_types': self._decode_records(atom_types) if atom_types is not None else None
        }
    
    def get_token_weights(self, freq_dict: Dict[str, float], temperature: float = 1.0, cutoff_token: str=None) -> Dict[str, float]:
        """Get balanced weights for each token in the vocabulary.
        
        Args:
            freq_dict: Dictionary mapping tokens to their frequencies
            temperature: Factor to scale frequency differences. Range (0, inf).
                temperature -> 0: weights become uniform
                temperature = 1: standard inverse frequency weights
                temperature > 1: amplifies differences between rare/common tokens
            cutoff_token: If specified, no token will have a higher weight than this one, even if more rare.
                
        Returns:
            Dictionary mapping tokens to weight values that average to 1.0
        """
        token_to_weight = compute_balanced_atom_weights_from_frequencies(freq_dict, temperature)
        if cutoff_token is not None:
            cutoff_weight = token_to_weight[cutoff_token]
            for token, weight in token_to_weight.items():
                if weight > cutoff_weight:
                    token_to_weight[token] = cutoff_weight
            # reweigh to MEAN 1 (not sum 1)
            mean_weight = np.mean(list(token_to_weight.values()))
            for token, weight in token_to_weight.items():
                token_to_weight[token] = weight / mean_weight

        weights = torch.ones(self.atom_vocab.vocab_size)
        for token, weight in token_to_weight.items():
            if token in self.atom_vocab.stoi:
                weights[self.atom_vocab.stoi[token]] = weight

        return weights, token_to_weight
