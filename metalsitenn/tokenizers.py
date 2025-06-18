# metalsitenn/tokenizers.py
'''
* Author: Evan Komp
* Created: 6/17/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import Union, List, Dict, Optional, Any
import warnings
import numpy as np

from metalsitenn.constants import COMMON_PROTEIN_ATOMS, ALL_METALS, BIOLOGICAL_METALS, PROTEIN_METALS, CRITICAL_METALS

class Tokenizer:
    """
    A generalizable tokenizer that maps between items and integer indices.
    
    Supports both vocabulary-based tokenization (from a list of items) and 
    index-based tokenization (from a specified vocabulary size).
    """
    
    def __init__(self, 
                 vocab: Union[List[Any], int], 
                 error_on_unknown: bool = True,
                 use_mask: bool = False):
        """
        Initialize the tokenizer.
        
        Args:
            vocab: Either a list of vocabulary items or an integer specifying vocab size
            error_on_unknown: If True, raise error on unknown tokens. If False, map to <UNK>
            use_mask: If True, add a <MASK> token to the vocabulary
        """
        self.error_on_unknown = error_on_unknown
        self.use_mask = use_mask
        self.special_tokens = []
        
        # Initialize mappings
        self.d2i = {}  # item to index mapping
        self.i2d = {}  # index to item mapping
        
        if isinstance(vocab, int):
            self._init_from_size(vocab)
        elif isinstance(vocab, (list, tuple)):
            self._init_from_list(vocab)
        else:
            raise ValueError("vocab must be either an integer or a list/tuple of items")
    
    def _init_from_size(self, size: int):
        """Initialize tokenizer from vocabulary size (items are indices 0 to size-1)."""
        if size <= 0:
            raise ValueError("Vocabulary size must be positive")
            
        # Add special tokens first
        current_idx = 0
        
        if not self.error_on_unknown:
            self.d2i['<UNK>'] = current_idx
            self.i2d[current_idx] = '<UNK>'
            self.special_tokens.append('<UNK>')
            current_idx += 1
            
        if self.use_mask:
            self.d2i['<MASK>'] = current_idx
            self.i2d[current_idx] = '<MASK>'
            self.special_tokens.append('<MASK>')
            current_idx += 1
        
        # Add regular tokens (0 to size-1, but adjusted for special tokens)
        remaining_size = size - current_idx
        if remaining_size <= 0:
            raise ValueError(f"Vocabulary size {size} too small for special tokens")
            
        for i in range(remaining_size):
            token = i  # Token is just the index
            self.d2i[token] = current_idx + i
            self.i2d[current_idx + i] = token
            
        self._original_vocab_size = size
        
    def _init_from_list(self, vocab_list: List[Any]):
        """Initialize tokenizer from a list of vocabulary items."""
        if len(vocab_list) == 0:
            raise ValueError("Vocabulary list cannot be empty")
            
        # Check for duplicates
        if len(set(vocab_list)) != len(vocab_list):
            warnings.warn("Vocabulary contains duplicates, only first occurrence will be kept")
            vocab_list = list(dict.fromkeys(vocab_list))  # Remove duplicates, preserve order
        
        current_idx = 0
        
        # Add special tokens first
        if not self.error_on_unknown:
            if '<UNK>' in vocab_list:
                warnings.warn("<UNK> found in vocabulary but error_on_unknown=False")
            self.d2i['<UNK>'] = current_idx
            self.i2d[current_idx] = '<UNK>'
            self.special_tokens.append('<UNK>')
            current_idx += 1
            
        if self.use_mask:
            if '<MASK>' in vocab_list:
                warnings.warn("<MASK> found in vocabulary but use_mask=True")
            self.d2i['<MASK>'] = current_idx
            self.i2d[current_idx] = '<MASK>'
            self.special_tokens.append('<MASK>')
            current_idx += 1
        
        # Add vocabulary items
        for item in vocab_list:
            if item not in ['<UNK>', '<MASK>']:  # Skip if already added as special token
                self.d2i[item] = current_idx
                self.i2d[current_idx] = item
                current_idx += 1
                
        self._original_vocab_size = len(vocab_list)
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.d2i)
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """Index of the <UNK> token, None if not present."""
        return self.d2i.get('<UNK>')
    
    @property
    def mask_token_id(self) -> Optional[int]:
        """Index of the <MASK> token, None if not present."""
        return self.d2i.get('<MASK>')
        
    def encode(self, item: Any, **kwargs) -> int:
        """
        Convert an item to its integer index.
        
        Args:
            item: Item to encode
            
        Returns:
            Integer index of the item
            
        Raises:
            KeyError: If item is unknown and error_on_unknown=True
        """
            
        if item in self.d2i:
            return self.d2i[item]
        
        if self.error_on_unknown:
            raise KeyError(f"Unknown item: {item}")
        else:
            return self.d2i['<UNK>']
    
    def decode(self, index: int) -> Any:
        """
        Convert an integer index to its corresponding item.
        
        Args:
            index: Integer index to decode
            
        Returns:
            Item corresponding to the index
            
        Raises:
            KeyError: If index is not in vocabulary
        """
        if index not in self.i2d:
            raise KeyError(f"Unknown index: {index}")
        return self.i2d[index]
    
    def encode_sequence(self, sequence: List[Any], **kwargs) -> List[int]:
        """
        Encode a sequence of items to a list of indices.
        
        Args:
            sequence: List of items to encode
            
        Returns:
            List of integer indices
        """
        return [self.encode(item, **kwargs) for item in sequence]
    
    def decode_sequence(self, indices: List[int]) -> List[Any]:
        """
        Decode a sequence of indices to a list of items.
        
        Args:
            indices: List of integer indices to decode
            
        Returns:
            List of items
        """
        return [self.decode(idx) for idx in indices]
    
    def mask_sequence(self, sequence: List[int], should_mask: List[bool]) -> List[int]:
        """
        Mask a sequence of indices based on a boolean mask.
        
        Args:
            sequence: List of integer indices to mask
            should_mask: List of booleans indicating which indices to mask
            
        Returns:
            List of masked indices, replacing masked items with <MASK> token
        """
        if self.mask_token_id is None:
            raise ValueError("Masking not supported, <MASK> token not defined")
        
        return [self.mask_token_id if mask else idx for idx, mask in zip(sequence, should_mask)]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __contains__(self, item: Any) -> bool:
        """Check if item is in vocabulary."""
        return item in self.d2i or (isinstance(item, int) and item in self.i2d)
    
    def __repr__(self) -> str:
        """Return string representation of tokenizer."""
        return (f"Tokenizer(vocab_size={self.vocab_size}, "
                f"error_on_unknown={self.error_on_unknown}, "
                f"use_mask={self.use_mask}, "
                f"special_tokens={self.special_tokens})")
    
    def get_vocab(self) -> Dict[Any, int]:
        """Return a copy of the item-to-index mapping."""
        return self.d2i.copy()
    
    def save_vocab(self, filepath: str):
        """
        Save vocabulary to a text file.
        
        Args:
            filepath: Path to save vocabulary file
        """
        with open(filepath, 'w') as f:
            for item, idx in sorted(self.d2i.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{item}\n")
    
    @classmethod
    def from_vocab_file(cls, filepath: str, error_on_unknown: bool = True, use_mask: bool = False):
        """
        Load tokenizer from a vocabulary file.
        
        Args:
            filepath: Path to vocabulary file (format: "index\titem" per line)
            error_on_unknown: If True, raise error on unknown tokens
            use_mask: If True, expect <MASK> token in vocabulary
            
        Returns:
            Tokenizer instance
        """
        vocab_items = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        idx, item = parts
                        # Try to convert item back to int if it was originally an int
                        try:
                            item = int(item)
                        except ValueError:
                            pass  # Keep as string
                        vocab_items.append((int(idx), item))
        
        # Sort by index and extract items
        vocab_items.sort(key=lambda x: x[0])
        vocab_list = [item for _, item in vocab_items]
        
        return cls(vocab_list, error_on_unknown, use_mask)
    

class ElementTokenizer(Tokenizer):
    """
    Specialized tokenizer for chemical elements commonly found in proteins.
    
    Includes common protein atoms, specific metals, and special tokens for unknown elements,
    masking, and generic metal representation.
    """
    
    def __init__(self, full_context_metals=None):
        """
        Initialize ElementTokenizer with protein atoms and selected metals vocabulary.
        
        Args:
            full_context_metals: Iterable of metal symbols to include explicitly in vocabulary.
                               Defaults to BIOLOGICAL_METALS | PROTEIN_METALS.
                               Other metals from ALL_METALS will be mapped to <METAL> token.
        
        The vocabulary includes:
        - Common protein atoms (C, N, O, H, S, P, etc.)
        - Specified metal elements (full_context_metals)
        - Special tokens: <UNK>, <MASK>, <METAL>
        """
        # Set default full_context_metals if not provided
        if full_context_metals is None:
            full_context_metals = BIOLOGICAL_METALS | PROTEIN_METALS | CRITICAL_METALS
        
        # Store the sets for later use
        self.full_context_metals = set(full_context_metals)
        self.other_metals = ALL_METALS - self.full_context_metals
        
        # Combine common protein atoms and full_context_metals into vocabulary
        vocab_elements = sorted(list(COMMON_PROTEIN_ATOMS | self.full_context_metals))
        
        # Initialize parent tokenizer with error_on_unknown=False and use_mask=True
        super().__init__(vocab_elements, error_on_unknown=False, use_mask=True)
        
        # Add <METAL> token after initialization
        metal_token_idx = len(self.d2i)
        self.d2i['<METAL>'] = metal_token_idx
        self.i2d[metal_token_idx] = '<METAL>'
        self.special_tokens.append('<METAL>')
    
    @property
    def metal_token_id(self) -> int:
        """Index of the <METAL> token."""
        return self.d2i['<METAL>']
    
    def encode(self, item: Any, metal_unknown: bool = False, **kwargs) -> int:
        """
        Convert an element to its integer index.
        
        Args:
            item: Element symbol to encode
            metal_unknown: If True, convert full_context_metals to <METAL> token instead of specific metal
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            Integer index of the element or special token
        """
        
        # If item is a metal not in full_context_metals, always map to <METAL>
        if item in self.other_metals:
            return self.d2i['<METAL>']
        
        # If metal_unknown is True and the item is in full_context_metals, return <METAL> token
        if metal_unknown and item in self.full_context_metals:
            return self.d2i['<METAL>']
        
        # Otherwise use parent encode method
        return super().encode(item, **kwargs)
    
    def encode_sequence(self, sequence: List[Any], metal_unknown: bool = False, **kwargs) -> List[int]:
        """
        Encode a sequence of elements to a list of indices.
        
        Args:
            sequence: List of element symbols to encode
            metal_unknown: If True, convert full_context_metals to <METAL> token instead of specific metals
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            List of integer indices
        """
        return [self.encode(item, metal_unknown=metal_unknown, **kwargs) for item in sequence]
    
    def is_metal(self, item: Any) -> bool:
        """
        Check if an element is a metal (from ALL_METALS).
        
        Args:
            item: Element symbol or index to check
            
        Returns:
            True if the element is a metal, False otherwise
        """
        if isinstance(item, int):
            if item in self.i2d:
                element = self.i2d[item]
                return element in ALL_METALS or element == '<METAL>'
            return False
        return item in ALL_METALS
    
    def is_full_context_metal(self, item: Any) -> bool:
        """
        Check if an element is a full context metal (explicitly in vocabulary).
        
        Args:
            item: Element symbol or index to check
            
        Returns:
            True if the element is a full context metal, False otherwise
        """
        if isinstance(item, int):
            if item in self.i2d:
                element = self.i2d[item]
                return element in self.full_context_metals
            return False
        return item in self.full_context_metals
    
    def is_other_metal(self, item: Any) -> bool:
        """
        Check if an element is an other metal (maps to <METAL> token).
        
        Args:
            item: Element symbol or index to check
            
        Returns:
            True if the element is an other metal, False otherwise
        """
        if isinstance(item, int):
            if item in self.i2d:
                element = self.i2d[item]
                return element in self.other_metals
            return False
        return item in self.other_metals
    
    def is_common_protein_atom(self, item: Any) -> bool:
        """
        Check if an element is a common protein atom.
        
        Args:
            item: Element symbol or index to check
            
        Returns:
            True if the element is a common protein atom, False otherwise
        """
        if isinstance(item, int):
            if item in self.i2d:
                element = self.i2d[item]
                return element in COMMON_PROTEIN_ATOMS
            return False
        return item in COMMON_PROTEIN_ATOMS
    
    def __repr__(self) -> str:
        """Return string representation of ElementTokenizer."""
        return (f"ElementTokenizer(vocab_size={self.vocab_size}, "
                f"n_protein_atoms={len(COMMON_PROTEIN_ATOMS)}, "
                f"n_full_context_metals={len(self.full_context_metals)}, "
                f"n_other_metals={len(self.other_metals)}, "
                f"special_tokens={self.special_tokens})")
        """Return string representation of ElementTokenizer."""
    

class ChargeTokenizer(Tokenizer):
    """
    Specialized tokenizer for atomic charges commonly found in molecular systems.
    
    Handles integer charge values from -3 to +3, with special tokens for masking.
    Charges outside the range are clamped to the nearest boundary.
    """
    
    def __init__(self):
        """
        Initialize ChargeTokenizer with charge values from -3 to +3.
        
        The vocabulary includes:
        - Integer charges from -3 to +3 (7 values)
        - Special token: <MASK>
        Total vocabulary size: 8
        """
        # Create vocabulary of charge values from -3 to +3
        charge_vocab = list(range(-3, 4))  # [-3, -2, -1, 0, 1, 2, 3]
        
        # Initialize parent tokenizer with error_on_unknown=True and use_mask=True
        super().__init__(charge_vocab, error_on_unknown=True, use_mask=True)
    
    def clamp_charge(self, charge: int) -> int:
        """
        Clamp charge value to the valid range [-3, 3].
        
        Args:
            charge: Integer charge value to clamp
            
        Returns:
            Clamped charge value within [-3, 3]
        """
        return max(-3, min(3, charge))
    
    def encode(self, item, clamp: bool = True, **kwargs) -> int:
        """
        Convert a charge value to its integer index.
        
        Args:
            item: Charge value to encode (int or float)
            clamp: If True, clamp out-of-range charges to [-3, 3]. If False, raise error.
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            Integer index of the charge value or special token
            
        Raises:
            ValueError: If charge is out of range and clamp=False
            TypeError: If item cannot be converted to integer
        """
        
        # Handle special tokens
        if item == '<MASK>' or (isinstance(item, str) and item in self.d2i):
            return super().encode(item, **kwargs)
        
        # Convert to integer charge
        try:
            charge = int(item)
        except (ValueError, TypeError):
            raise TypeError(f"Cannot convert {item} to integer charge")
        
        # Clamp or validate charge range
        if clamp:
            charge = self.clamp_charge(charge)
        else:
            if charge < -3 or charge > 3:
                raise ValueError(f"Charge {charge} is outside valid range [-3, 3]")
        
        # Use parent encode method
        return super().encode(charge, **kwargs)
    
    def encode_sequence(self, sequence, clamp: bool = True, **kwargs) -> list[int]:
        """
        Encode a sequence of charge values to a list of indices.
        
        Args:
            sequence: List of charge values to encode
            clamp: If True, clamp out-of-range charges to [-3, 3]. If False, raise error.
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            List of integer indices
        """
        return [self.encode(item, clamp=clamp, **kwargs) for item in sequence]
    
    def is_positive(self, item) -> bool:
        """
        Check if a charge value (or its index) is positive.
        
        Args:
            item: Charge value or index to check
            
        Returns:
            True if the charge is positive (> 0), False otherwise
        """
        if isinstance(item, int) and item in self.i2d:
            charge = self.i2d[item]
            if isinstance(charge, int):
                return charge > 0
            return False
        return item > 0 if isinstance(item, (int, float)) else False
    
    def __repr__(self) -> str:
        """Return string representation of ChargeTokenizer."""
        return (f"ChargeTokenizer(vocab_size={self.vocab_size}, "
                f"charge_range=[-3, 3], "
                f"special_tokens={self.special_tokens})")
    
class HydrogenCountTokenizer(Tokenizer):
    """
    Specialized tokenizer for hydrogen counts from 0 to 4.
    """
    
    def __init__(self):
        """Initialize HydrogenCountTokenizer with hydrogen counts 0-4 and <MASK> token."""
        hydrogen_vocab = list(range(0, 5))  # [0, 1, 2, 3, 4]
        super().__init__(hydrogen_vocab, error_on_unknown=True, use_mask=True)


class HybdridizaitonTokenizer(Tokenizer):
    """
    Specialized tokenizer for hybridization states.
    
    Supports:
    sp, sp2, sp3, square planar, trigonal bipyramidal, octahedral

    Which will be coming in as numbers 0-5
    """

    def __init__(self):
        """Initialize HybridizationTokenizer with hybridization states and <MASK> token."""
        hybridization_vocab = list(range(0, 6))  # [0, 1, 2, 3, 4, 5]
        super().__init__(hybridization_vocab, error_on_unknown=True, use_mask=True)


class BondOrderTokenizer(Tokenizer):
    """
    Specialized tokenizer for bond orders commonly found in molecular systems.
    
    Handles integer bond orders from 1 to 4 (single, double, triple, quadruple),
    with special token for masking. Only accepts known bond orders.
    """
    
    def __init__(self):
        """
        Initialize BondOrderTokenizer with bond orders 1-4 and <MASK> token.
        
        The vocabulary includes:
        - Bond orders 0, 1, 2, 3, 4
        - Special token: <MASK>
        Total vocabulary size: 6
        """
        # Create vocabulary of bond orders from 1 to 4
        bond_order_vocab = list(range(0, 5))  # [1, 2, 3, 4]
        
        # Initialize parent tokenizer with error_on_unknown=True and use_mask=True
        super().__init__(bond_order_vocab, error_on_unknown=True, use_mask=True)
    
    @property
    def non_bonded_token_id(self) -> int:
        """Index of the 0 (non-bonded) token."""
        return self.d2i[0]
    
    def __repr__(self) -> str:
        """Return string representation of BondOrderTokenizer."""
        return (f"BondOrderTokenizer(vocab_size={self.vocab_size}, "
                f"bond_orders=[1,2,3,4], "
                f"special_tokens={self.special_tokens})")


class AromaticTokenizer(Tokenizer):
    """
    Specialized tokenizer for aromatic bond status.
    
    Handles boolean aromatic status (True/False) with special token for masking.
    Only accepts known values.
    """
    
    def __init__(self):
        """
        Initialize AromaticTokenizer with aromatic status values and <MASK> token.
        
        The vocabulary includes:
        - False (not aromatic)
        - True (aromatic)
        - Special token: <MASK>
        Total vocabulary size: 3
        """
        # Create vocabulary of aromatic status
        aromatic_vocab = [False, True]
        
        # Initialize parent tokenizer with error_on_unknown=True and use_mask=True
        super().__init__(aromatic_vocab, error_on_unknown=True, use_mask=True)
    
    @property
    def non_bonded_token_id(self) -> int:
        """Index of the False (not aromatic) token."""
        return self.d2i[False]
    
    def __repr__(self) -> str:
        """Return string representation of AromaticTokenizer."""
        return (f"AromaticTokenizer(vocab_size={self.vocab_size}, "
                f"aromatic_values=[False,True], "
                f"special_tokens={self.special_tokens})")


class RingTokenizer(Tokenizer):
    """
    Specialized tokenizer for ring bond membership status.
    
    Handles boolean ring membership (True/False) with special token for masking.
    Only accepts known values.
    """
    
    def __init__(self):
        """
        Initialize RingTokenizer with ring membership values and <MASK> token.
        
        The vocabulary includes:
        - False (not in ring)
        - True (in ring)
        - Special token: <MASK>
        Total vocabulary size: 3
        """
        # Create vocabulary of ring membership status
        ring_vocab = [False, True]
        
        # Initialize parent tokenizer with error_on_unknown=True and use_mask=True
        super().__init__(ring_vocab, error_on_unknown=True, use_mask=True)
    
    @property
    def non_bonded_token_id(self) -> int:
        """Index of the False (not in ring) token."""
        return self.d2i[False]
    
    def __repr__(self) -> str:
        """Return string representation of RingTokenizer."""
        return (f"RingTokenizer(vocab_size={self.vocab_size}, "
                f"ring_values=[False,True], "
                f"special_tokens={self.special_tokens})")
    

class BondDistanceTokenizer(Tokenizer):
    """
    Specialized tokenizer for bond distances.
    
    This is distance in the number of bonds between sense, not angstrom sense.
    If distance is > 7 or in a differnt graph, it will be assigned the same token
    """

    def __init__(self):
        distance_vocab = list(range(0, 8))  # [0, 1, 2, 3, 4, 5, 6, 7]
        super().__init__(distance_vocab, error_on_unknown=True, use_mask=True)


    def encode(self, item: Any, **kwargs) -> int:
        """
        Convert a bond distance value to its integer index.
        
        Args:
            item: Bond distance value to encode (int or float)
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            Integer index of the bond distance value or special token
            
        Raises:
            ValueError: If distance is out of range and error_on_unknown=True
            TypeError: If item cannot be converted to integer
        """
        if item > 7:
            item =0
        # Use parent encode method
        return super().encode(item, **kwargs)
    
    @property
    def non_bonded_token_id(self) -> int:
        """Index of the 0 (non-bonded) token."""
        return self.d2i[0]


TOKENIZERS = {
    'element': ElementTokenizer(),
    'charge': ChargeTokenizer(),
    'nhyd': HydrogenCountTokenizer(),
    'hyb': HybdridizaitonTokenizer(),
    'bond_order': BondOrderTokenizer(),
    'is_aromatic': AromaticTokenizer(),
    'is_in_ring': RingTokenizer(),
    'bond_distance': BondDistanceTokenizer(),
}
    