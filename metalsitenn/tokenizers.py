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
import torch

from metalsitenn.constants import (COMMON_PROTEIN_ATOMS, ALL_METALS, BIOLOGICAL_METALS, PROTEIN_METALS, CRITICAL_METALS,
    am3d_ALKAILI_METALS, am3d_MG, am3d_CA, am3d_ZN, am3d_NON_ZN_TM, am3d_UNCOVERED_METAL)

import logging
logger = logging.getLogger(__name__)

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
    

########################################################################
######################################################################## ELEMENT TOKENIZER
########################################################################
UNCOMMON_METALS = ALL_METALS - PROTEIN_METALS - BIOLOGICAL_METALS
am3d_AGGREGATOR = {}
am3d_AGGREGATOR.update(dict(zip(am3d_ALKAILI_METALS, ['<am3d_ALKALI>'] * len(am3d_ALKAILI_METALS))))
am3d_AGGREGATOR.update(dict(zip(am3d_MG, ['<am3d_MG>'] * len(am3d_MG))))
am3d_AGGREGATOR.update(dict(zip(am3d_CA, ['<am3d_CA>'] * len(am3d_CA))))
am3d_AGGREGATOR.update(dict(zip(am3d_ZN, ['<am3d_ZN>'] * len(am3d_ZN))))
am3d_AGGREGATOR.update(dict(zip(am3d_NON_ZN_TM, ['<am3d_NON_ZN_TM>'] * len(am3d_NON_ZN_TM))))
am3d_AGGREGATOR.update(dict(zip(am3d_UNCOVERED_METAL, ['<METAL>'] * len(am3d_UNCOVERED_METAL))))

DEFAULT_AGGREGATORS = {
    'unknown_metal': dict(zip(ALL_METALS, ['<METAL>'] * len(ALL_METALS))),
    'uncommon_metal': dict(zip(UNCOMMON_METALS, ['<METAL>'] * len(UNCOMMON_METALS))),
    'allmetal3d_groups': am3d_AGGREGATOR}

class ElementTokenizer(Tokenizer):
    """
    Specialized tokenizer for chemical elements with configurable aggregation schemes.
    
    Supports flexible element grouping through named aggregation mappings that can be
    activated at encode-time. Vocabulary is built to include both individual elements
    and aggregation target tokens.
    """
    
    def __init__(self, 
                 base_vocabulary=None,
                 aggregation_schemes=None,
                 custom_aggregation_schemes=None):
        """
        Initialize ElementTokenizer with configurable element vocabulary and aggregation.
        
        Args:
            base_vocabulary: Iterable of element symbols to include in vocabulary.
                           Defaults to COMMON_PROTEIN_ATOMS | ALL_METALS.
                           These elements can still be aggregated if aggregators are active.
            aggregation_schemes: Dict of named aggregation mappings to use.
                               If None, uses DEFAULT_AGGREGATORS.
            custom_aggregation_schemes: Additional user-defined aggregation schemes
                                      to merge with default/provided schemes.
        
        The vocabulary includes:
        - Base vocabulary elements (can be aggregated or not depending on active aggregators)
        - All aggregation target tokens from all schemes
        - Special tokens: <UNK>, <MASK>
        """
        # Set default base_vocabulary if not provided
        if base_vocabulary is None:
            base_vocabulary = COMMON_PROTEIN_ATOMS | ALL_METALS
        
        self.base_vocabulary = set(base_vocabulary)
        
        # Setup aggregation schemes
        self.aggregation_schemes = aggregation_schemes or DEFAULT_AGGREGATORS.copy()
        if custom_aggregation_schemes:
            self.aggregation_schemes.update(custom_aggregation_schemes)
        
        # Validate aggregation schemes
        self._validate_aggregation_schemes()
        
        # Build vocabulary including all possible tokens
        vocab_elements = self._build_vocabulary()
        
        # Initialize parent tokenizer
        super().__init__(vocab_elements, error_on_unknown=False, use_mask=True)
        
        # Cache aggregation mappings for efficiency
        self._cache_aggregation_data()
    
    def _validate_aggregation_schemes(self):
        """
        Validate that aggregation scheme keys are valid element symbols.
        
        Raises:
            ValueError: If any aggregation key is not a valid element symbol
        """
        # Assume we have access to ALL_ELEMENTS set containing valid element symbols
        for scheme_name, mapping in self.aggregation_schemes.items():
            invalid_elements = []
            for element in mapping.keys():
                # Check against known element sets (you may need to adjust this validation)
                if (element not in ALL_METALS and 
                    element not in COMMON_PROTEIN_ATOMS and
                    not element.isupper() or len(element) > 2):  # Basic element symbol check
                    invalid_elements.append(element)
            
            if invalid_elements:
                raise ValueError(f"Aggregation scheme '{scheme_name}' contains invalid "
                               f"element symbols: {invalid_elements}")
    
    def _build_vocabulary(self):
        """
        Build vocabulary including base elements and all aggregation targets.
        
        Returns:
            Sorted list of vocabulary elements including aggregation tokens
        """
        vocab_elements = set(self.base_vocabulary)
        
        # Add all unique aggregation target tokens
        for scheme_name, mapping in self.aggregation_schemes.items():
            vocab_elements.update(mapping.values())
        
        # Convert to sorted list for consistent ordering
        return sorted(list(vocab_elements))
    
    def _cache_aggregation_data(self):
        """Cache aggregation mappings for efficient lookup during encoding."""
        # Create reverse mapping: element -> list of (scheme_name, target_token) pairs
        self._element_to_aggregations = {}
        
        for scheme_name, mapping in self.aggregation_schemes.items():
            for element, target_token in mapping.items():
                if element not in self._element_to_aggregations:
                    self._element_to_aggregations[element] = []
                self._element_to_aggregations[element].append((scheme_name, target_token))
        
        # Cache all aggregation target tokens for quick identification
        self._aggregation_tokens = set()
        for mapping in self.aggregation_schemes.values():
            self._aggregation_tokens.update(mapping.values())
        
        # Build mapping: token -> set of original elements that can produce it
        self._token_to_original_elements = {}
        
        # First, add direct mappings (tokens that are elements themselves)
        for element in self.base_vocabulary:
            if element in self.d2i:
                token_id = self.d2i[element]
                self._token_to_original_elements[token_id] = {element}
        
        # Then add aggregation mappings
        for scheme_name, mapping in self.aggregation_schemes.items():
            for original_element, target_token in mapping.items():
                if target_token in self.d2i:
                    token_id = self.d2i[target_token]
                    if token_id not in self._token_to_original_elements:
                        self._token_to_original_elements[token_id] = set()
                    self._token_to_original_elements[token_id].add(original_element)
        
        # Cache which tokens can represent metals
        self._token_can_be_metal = {}
        for token_id, original_elements in self._token_to_original_elements.items():
            # Token can represent metal if any of its original elements are metals
            self._token_can_be_metal[token_id] = any(elem in ALL_METALS for elem in original_elements)

    @property
    def _str_can_be_metal(self) -> bool:
        out = {}
        for tok, can_be_metal in self._token_can_be_metal:
            if can_be_metal:
                out[self.i2d[tok]] = True
            else:
                out[self.i2d[tok]] = False
        return out
    
    def encode(self, item: Any, active_aggregators: List[str] = None, **kwargs) -> int:
        """
        Convert an element to its integer index, applying active aggregation schemes.
        
        Args:
            item: Element symbol to encode
            active_aggregators: List of aggregation scheme names to apply in order.
                               First matching scheme takes precedence.
            **kwargs: Additional keyword arguments for compatibility
            
        Returns:
            Integer index of the element or aggregated token
            
        Raises:
            ValueError: If any active_aggregator is not a known scheme
        """
        if active_aggregators is None:
            active_aggregators = []
        
        # Validate active aggregators
        unknown_aggregators = [agg for agg in active_aggregators 
                             if agg not in self.aggregation_schemes]
        if unknown_aggregators:
            raise ValueError(f"Unknown aggregation schemes: {unknown_aggregators}. "
                           f"Available schemes: {list(self.aggregation_schemes.keys())}")
        
        # Apply aggregation schemes in order
        if item in self._element_to_aggregations:
            for scheme_name, target_token in self._element_to_aggregations[item]:
                if scheme_name in active_aggregators:
                    # Found matching active aggregator - use target token
                    return super().encode(target_token, **kwargs)
        
        # No aggregation applied - use standard encoding
        return super().encode(item, **kwargs)
    
    def encode_sequence(self, sequence: List[Any], 
                       active_aggregators: List[str] = None, **kwargs) -> List[int]:
        """
        Encode a sequence of elements with consistent aggregation.
        
        Args:
            sequence: List of element symbols to encode
            active_aggregators: List of aggregation scheme names to apply
            **kwargs: Additional keyword arguments for compatibility
            
        Returns:
            List of integer indices
        """
        return [self.encode(item, active_aggregators=active_aggregators, **kwargs) 
                for item in sequence]
    
    def is_aggregation_token(self, item: Any) -> bool:
        """
        Check if an item is an aggregation target token.
        
        Args:
            item: Element symbol or token index to check
            
        Returns:
            True if the item is an aggregation target token
        """
        if isinstance(item, int):
            if item in self.i2d:
                token = self.i2d[item]
                return token in self._aggregation_tokens
            return False
        return item in self._aggregation_tokens
    
    def is_base_vocabulary_element(self, item: Any) -> bool:
        """
        Check if an element is in the base vocabulary.
        
        Args:
            item: Element symbol or token index to check
            
        Returns:
            True if the element is in base vocabulary
        """
        if isinstance(item, int):
            if item in self.i2d:
                element = self.i2d[item]
                return element in self.base_vocabulary
            return False
        return item in self.base_vocabulary
    
    def get_aggregation_mapping(self, scheme_name: str) -> Dict[str, str]:
        """
        Get the mapping for a specific aggregation scheme.
        
        Args:
            scheme_name: Name of the aggregation scheme
            
        Returns:
            Dictionary mapping elements to target tokens
            
        Raises:
            KeyError: If scheme_name is not found
        """
        if scheme_name not in self.aggregation_schemes:
            raise KeyError(f"Unknown aggregation scheme: {scheme_name}")
        return self.aggregation_schemes[scheme_name].copy()
    
    def get_available_aggregators(self) -> List[str]:
        """Get list of available aggregation scheme names."""
        return list(self.aggregation_schemes.keys())
    
    def would_be_aggregated(self, element: str, active_aggregators: List[str]) -> bool:
        """
        Check if an element would be aggregated given active aggregation schemes.
        
        Args:
            element: Element symbol to check
            active_aggregators: List of active aggregation scheme names
            
        Returns:
            True if element would be aggregated, False otherwise
        """
        if element in self._element_to_aggregations:
            for scheme_name, _ in self._element_to_aggregations[element]:
                if scheme_name in active_aggregators:
                    return True
        return False
    
    def get_aggregation_target(self, element: str, active_aggregators: List[str]) -> str:
        """
        Get the target token for an element given active aggregation schemes.
        
        Args:
            element: Element symbol to check
            active_aggregators: List of active aggregation scheme names
            
        Returns:
            Target token if aggregated, otherwise the original element
        """
        if element in self._element_to_aggregations:
            for scheme_name, target_token in self._element_to_aggregations[element]:
                if scheme_name in active_aggregators:
                    return target_token
        return element
    
    @property
    def is_metal_token(self, item) -> bool:
        """
        Check if the tokenizer can represent metal elements.
        
        Returns:
            True if any token can represent a metal, False otherwise
        """
        if isinstance(item, int):
            return self._token_can_be_metal.get(item, False)
        elif isinstance(item, str):
            # Check if item is a metal in the base vocabulary
            return self._str_can_be_metal.get(item, False)
        return False
    
    def can_token_represent_metal(self, token_id: int) -> bool:
        """
        Check if a token ID can potentially represent a metal (under any aggregation).
        
        Args:
            token_id: Token ID to check
            
        Returns:
            True if token can represent metals under some aggregation scheme
        """
        return self._token_can_be_metal.get(token_id, False)
    
    def get_original_elements_for_token(self, token_id: int) -> set:
        """
        Get the set of original elements that can produce this token.
        
        Args:
            token_id: Token ID to look up
            
        Returns:
            Set of original element symbols that can produce this token
        """
        return self._token_to_original_elements.get(token_id, set())
    
    def get_metal_representing_token_ids(self, active_aggregators: List[str] = None) -> List[int]:
        """
        Get token IDs that represent metals given active aggregation schemes.
        
        Args:
            active_aggregators: List of active aggregation scheme names
            
        Returns:
            Sorted list of token IDs that can represent metals
        """
        if active_aggregators is None:
            active_aggregators = []
        
        metal_token_ids = set()
        
        # Check each token to see if it represents metals under current aggregation
        for token_id in self._token_to_original_elements:
            if self._token_can_be_metal[token_id]:
                # This token CAN represent metals, but does it under current aggregation?
                
                # Get the token symbol
                token_symbol = self.i2d[token_id]
                
                # Case 1: Token is itself a metal and not aggregated away
                if token_symbol in ALL_METALS:
                    if not self.would_be_aggregated(token_symbol, active_aggregators):
                        metal_token_ids.add(token_id)
                
                # Case 2: Token is an aggregation target that receives metals
                elif token_symbol in self._aggregation_tokens:
                    # Check if any active aggregator maps metals to this token
                    for aggregator in active_aggregators:
                        if aggregator in self.aggregation_schemes:
                            mapping = self.aggregation_schemes[aggregator]
                            for original_elem, target_token in mapping.items():
                                if target_token == token_symbol and original_elem in ALL_METALS:
                                    metal_token_ids.add(token_id)
                                    break  # Found one, no need to check more
        
        return sorted(list(metal_token_ids))
    
    def count_metal_composition_from_tokens(self, 
                                           token_ids: List[int], 
                                           active_aggregators: List[str] = None,
                                           include_special_tokens: bool = False
                                           ) -> torch.Tensor:
        """
        Generate metal composition count vector from tokenized element IDs.
        
        Args:
            token_ids: List or tensor of tokenized element IDs (can include duplicates)
            active_aggregators: List of active aggregation scheme names used during encoding
            
        Returns:
            Count vector with counts for each metal-representing token
        """
        # Convert to tensor if not already
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Get metal-representing token IDs for current aggregation
        metal_token_ids = self.get_metal_representing_token_ids(active_aggregators)
        if include_special_tokens:
            metal_token_ids.insert(0, self.mask_token_id)  # Include <MASK> if requested
            metal_token_ids.insert(0, self.unk_token_id)  # Include <UNK> if requested

        metal_token_ids_tensor = torch.tensor(metal_token_ids, dtype=torch.long)
        
        # Initialize count vector
        composition = torch.zeros(len(metal_token_ids), dtype=torch.float)
        
        # Only count tokens that represent metals under current aggregation
        valid_mask = torch.isin(token_ids, metal_token_ids_tensor)
        valid_tokens = token_ids[valid_mask]
        
        if len(valid_tokens) > 0:
            # Map token IDs to positions in composition vector
            token_to_pos = {int(token_id): i for i, token_id in enumerate(metal_token_ids)}
            
            # Map valid tokens to their positions
            positions = torch.tensor([token_to_pos[int(token)] for token in valid_tokens], 
                                   dtype=torch.long)
            
            # Use bincount to count occurrences
            counts = torch.bincount(positions, minlength=len(metal_token_ids))
            composition = counts.float()
        
        return composition
    
    def get_metal_composition_labels(self, active_aggregators: List[str] = None) -> List[str]:
        """
        Get ordered labels for metal composition vector positions.
        
        Args:
            active_aggregators: List of active aggregation scheme names
            
        Returns:
            List of token symbols corresponding to composition vector positions
        """
        metal_token_ids = self.get_metal_representing_token_ids(active_aggregators)
        return [self.i2d[token_id] for token_id in metal_token_ids]
    
    def encode_metal_composition_from_elements(self, 
                                             elements: List[str], 
                                             active_aggregators: List[str] = None) -> torch.Tensor:
        """
        Generate metal composition count vector from original element symbols.
        
        Args:
            elements: List of original element symbols (can include duplicates)
            active_aggregators: List of active aggregation scheme names
            
        Returns:
            Count vector with counts for each metal-representing token
        """
        # First tokenize the elements with aggregation
        token_ids = self.encode_sequence(elements, active_aggregators=active_aggregators)
        
        # Then count metals from the tokens
        return self.count_metal_composition_from_tokens(token_ids, active_aggregators)
    
    def get_metal_vocab_size(self, active_aggregators: List[str] = None) -> int:
        """
        Get the size of metal vocabulary given active aggregation schemes.
        
        Args:
            active_aggregators: List of active aggregation scheme names
            
        Returns:
            Number of tokens that can represent metals under current aggregation
        """
        return len(self.get_metal_representing_token_ids(active_aggregators))
    
    def decode_metal_composition_counts(self, 
                                       count_vector: Union[torch.Tensor, List[float]], 
                                       active_aggregators: List[str] = None,
                                       threshold: float = 0.5) -> List[str]:
        """
        Decode metal composition count vector back to list of metal token symbols.
        
        Args:
            count_vector: Count vector for metal tokens (from count_metal_composition_from_tokens)
            active_aggregators: List of active aggregation scheme names used during encoding
            threshold: Minimum count threshold for including a token (default 0.5)
            
        Returns:
            List of metal token symbols, repeated according to their counts
            
        Example:
            count_vector = [1.0, 2.0, 0.0, 1.0]  # Fe=1, Zn=2, Ca=0, <METAL>=1
            labels = ['Fe', 'Zn', 'Ca', '<METAL>']
            returns: ['Fe', 'Zn', 'Zn', '<METAL>']
        """
        # Convert to list if tensor
        if isinstance(count_vector, torch.Tensor):
            count_vector = count_vector.tolist()
        
        # Get corresponding labels
        labels = self.get_metal_composition_labels(active_aggregators)
        
        if len(count_vector) != len(labels):
            raise ValueError(f"Count vector length ({len(count_vector)}) does not match "
                           f"number of metal labels ({len(labels)}) for active aggregators {active_aggregators}")
        
        # Build result list
        result = []
        for i, (count, label) in enumerate(zip(count_vector, labels)):
            if count > threshold:
                # Add the token repeated by its count (rounded to nearest int)
                repeat_count = int(round(count))
                result.extend([label] * repeat_count)
        
        return result
    
    def __repr__(self) -> str:
        """Return string representation of ElementTokenizer."""
        return (f"ElementTokenizer(vocab_size={self.vocab_size}, "
                f"n_base_elements={len(self.base_vocabulary)}, "
                f"n_aggregation_schemes={len(self.aggregation_schemes)}, "
                f"n_aggregation_tokens={len(self._aggregation_tokens)}, "
                f"special_tokens={self.special_tokens})")
    
    @property
    def metal_token_id(self) -> List[int]:
        return self.d2i.get('<METAL>', None)





########################################################################
########################################################################
########################################################################

    


    

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
        super().__init__(hydrogen_vocab, error_on_unknown=False, use_mask=True)


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
        super().__init__(hybridization_vocab, error_on_unknown=False, use_mask=True)


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
    



TOKENIZERS = {
    'element': ElementTokenizer(),
    'charge': ChargeTokenizer(),
    'nhyd': HydrogenCountTokenizer(),
    'hyb': HybdridizaitonTokenizer(),
    'bond_order': BondOrderTokenizer(),
    'is_aromatic': AromaticTokenizer(),
    'is_in_ring': RingTokenizer(),
}
    