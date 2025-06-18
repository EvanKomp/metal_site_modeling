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

from metalsitenn.tokenizers import TOKENIZERS
from metalsitenn.placer_modules.cifutils import Chain
from metalsitenn.constants import I2E


class MetalSiteFeaturizer:
    """
    Featurizer for metal binding sites that converts Chain objects to tokenized features.
    
    Processes both atom-level and bond-level features using the configured self.tokenizers.
    Atom features are returned as (N_atoms, 1) tensors, bond features as (N_atoms, N_atoms) matrices.
    """
    tokenizers = TOKENIZERS
    
    def __init__(self, atom_features: List[str]=['element'], bond_features: List[str]=['is_bonded']):
        """
        Initialize the featurizer with specified features.
        
        Args:
            atom_features: List of atom feature names (keys in self.tokenizers)
            bond_features: List of bond feature names (keys in self.tokenizers)
            
        Raises:
            ValueError: If any requested feature is not available in self.tokenizers
        """
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
    
    def __call__(self, chain: Chain, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Convert a Chain object to tokenized atom and bond features.
        
        Args:
            chain: Chain object representing a metal binding site
            **kwargs: Additional arguments passed to self.tokenizers
            
        Returns:
            Tuple of (atom_features_dict, bond_features_dict)
            - atom_features_dict: Dict with keys as feature names, values as (N_atoms, 1) tensors
                                 Always includes 'atom_resid' and 'atom_resname' as lists
            - bond_features_dict: Dict with keys as feature names, values as (N_atoms, N_atoms) tensors
            
        Raises:
            ValueError: If bonds reference atoms not present in chain.atoms
        """
        if len(chain.atoms) == 0:
            warnings.warn("Chain has no atoms, returning empty features")
            return {}, {}
        
        # Create consistent atom ordering and indexing
        atom_keys = list(chain.atoms.keys())
        atom_to_idx = {key: idx for idx, key in enumerate(atom_keys)}
        n_atoms = len(atom_keys)
        
        # Initialize feature dictionaries
        atom_features_dict = {}
        bond_features_dict = {}
        
        # Process atom features
        atom_features_dict.update(self._process_atom_features(chain, atom_keys, **kwargs))

        # add positions
        positions = torch.tensor([chain.atoms[key].xyz for key in atom_keys], dtype=torch.float32)
        # find the center of mass
        com = positions.mean(dim=0)
        positions -= com  # center the positions around the origin

        atom_features_dict['positions'] = positions
        
        # Process bond features
        if self.bond_features:
            bond_features_dict.update(self._process_bond_features(chain, atom_to_idx, n_atoms, **kwargs))
        
        return atom_features_dict, bond_features_dict
    
    def _process_atom_features(self, chain: Chain, atom_keys: List[Tuple], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Process atom-level features.
        
        Args:
            chain: Chain object
            atom_keys: Ordered list of atom keys
            **kwargs: Additional arguments passed to self.tokenizers
            
        Returns:
            Dictionary of atom features as (N_atoms, 1) tensors, plus residue info as lists
        """
        atom_features_dict = {}
        n_atoms = len(atom_keys)
        
        # Always include residue information
        atom_resids = []
        atom_resnames = []
        atom_name = []
        
        for atom_key in atom_keys:
            # atom_key is (chain_id, res_num, res_name, atom_name)
            atom_resids.append(atom_key[1])  # res_num
            atom_resnames.append(atom_key[2])  # res_name
            atom_name.append(atom_key[3])  # atom_name
        
        atom_features_dict['atom_resid'] = atom_resids
        atom_features_dict['atom_resname'] = atom_resnames
        atom_features_dict['atom_name'] = atom_name
        
        # Process requested atom features
        for feature_name in self.atom_features:
            tokenizer = self.tokenizers[feature_name]
            feature_values = []
            
            for atom_key in atom_keys:
                atom = chain.atoms[atom_key]
                
                # Extract the appropriate property based on feature name
                if feature_name == 'element':
                    # Convert atomic number to element symbol
                    element_symbol = I2E.get(atom.element, 'X')  # 'X' for unknown elements
                    value = tokenizer.encode(element_symbol, **kwargs)
                elif feature_name == 'charge':
                    value = tokenizer.encode(atom.charge, **kwargs)
                elif feature_name == 'nhyd':
                    value = tokenizer.encode(atom.nhyd, **kwargs)
                elif feature_name == 'hyb':
                    value = tokenizer.encode(atom.hyb, **kwargs)
                else:
                    raise ValueError(f"Unknown atom feature: {feature_name}")
                
                feature_values.append(value)
            
            # Convert to (N_atoms, 1) tensor
            atom_features_dict[feature_name] = torch.tensor(feature_values, dtype=torch.long).unsqueeze(1)
        
        return atom_features_dict
    
    def _process_bond_features(self, chain: Chain, atom_to_idx: Dict, n_atoms: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Process bond-level features.
        
        Args:
            chain: Chain object
            atom_to_idx: Mapping from atom keys to indices
            n_atoms: Number of atoms
            **kwargs: Additional arguments passed to self.tokenizers
            
        Returns:
            Dictionary of bond features as (N_atoms, N_atoms) tensors
        """
        bond_features_dict = {}
        
        # Validate that all bonds reference existing atoms
        for bond in chain.bonds:
            if bond.a not in atom_to_idx:
                raise ValueError(f"Bond references unknown atom: {bond.a}")
            if bond.b not in atom_to_idx:
                raise ValueError(f"Bond references unknown atom: {bond.b}")
        
        # Process each bond feature
        for feature_name in self.bond_features:
            tokenizer = self.tokenizers[feature_name]
            
            # Initialize matrix with non-bonded tokens
            bond_matrix = torch.full((n_atoms, n_atoms), tokenizer.non_bonded_token_id, dtype=torch.long)
            
            if feature_name == 'bond_distance':
                # Special handling for bond distance - need to compute shortest paths
                bond_matrix = self._compute_bond_distances(chain, atom_to_idx, n_atoms, tokenizer, **kwargs)
            else:
                # Process regular bond features
                for bond in chain.bonds:
                    i, j = atom_to_idx[bond.a], atom_to_idx[bond.b]
                    
                    # Extract the appropriate property based on feature name
                    if feature_name == 'is_bonded':
                        value = tokenizer.encode(True, **kwargs)
                    elif feature_name == 'bond_order':
                        value = tokenizer.encode(bond.order, **kwargs)
                    elif feature_name == 'is_aromatic':
                        value = tokenizer.encode(bond.aromatic, **kwargs)
                    elif feature_name == 'is_in_ring':
                        value = tokenizer.encode(bond.in_ring, **kwargs)
                    else:
                        raise ValueError(f"Unknown bond feature: {feature_name}")
                    
                    # Set symmetric values (bonds are undirected)
                    bond_matrix[i, j] = value
                    bond_matrix[j, i] = value
            
            bond_features_dict[feature_name] = bond_matrix
        
        return bond_features_dict
    
    def _compute_bond_distances(self, chain: Chain, atom_to_idx: Dict, n_atoms: int, 
                               tokenizer, **kwargs) -> torch.Tensor:
        """
        Compute bond distances using NetworkX shortest path algorithm.
        
        Args:
            chain: Chain object
            atom_to_idx: Mapping from atom keys to indices
            n_atoms: Number of atoms
            tokenizer: Bond distance tokenizer
            **kwargs: Additional arguments passed to tokenizer
            
        Returns:
            (N_atoms, N_atoms) tensor of tokenized bond distances
        """
        # Initialize matrix with non-bonded tokens (distance > 7 or disconnected)
        distance_matrix = torch.full((n_atoms, n_atoms), tokenizer.non_bonded_token_id, dtype=torch.long)
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_edges_from([(bond.a, bond.b) for bond in chain.bonds])
        
        # Compute all-pairs shortest path lengths with cutoff=7
        try:
            distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=7))
        except nx.NetworkXError:
            # Handle case where graph is empty
            return distance_matrix
        
        # Fill distance matrix
        for atom_a, paths in distances.items():
            if atom_a not in atom_to_idx:
                continue
            i = atom_to_idx[atom_a]
            
            for atom_b, distance in paths.items():
                if atom_b not in atom_to_idx:
                    continue
                j = atom_to_idx[atom_b]
                
                # Tokenize distance (tokenizer handles distance > 7 by setting to 0)
                tokenized_distance = tokenizer.encode(distance, **kwargs)
                distance_matrix[i, j] = tokenized_distance
        
        return distance_matrix
    
    def get_feature_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for all configured features.
        
        Returns:
            Dictionary mapping feature names to their vocabulary sizes
        """
        vocab_sizes = {}
        
        for feature_name in self.atom_features + self.bond_features:
            tokenizer = self.tokenizers[feature_name]
            vocab_sizes[feature_name] = tokenizer.vocab_size
        
        return vocab_sizes
    
    def __repr__(self) -> str:
        """Return string representation of the featurizer."""
        return (f"MetalSiteFeaturizer(atom_features={self.atom_features}, "
                f"bond_features={self.bond_features})")
    

    def mask_atoms(self, 
                atom_features_dict: Dict[str, torch.Tensor], 
                bond_features_dict: Dict[str, torch.Tensor],
                mask_prob: float = 0.15,
                mask_token_prob: float = 0.8,
                random_token_prob: float = 0.1,
                mask_bonds: bool = True,
                                random_seed: int = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply BERT-style masking to atom and bond features.
        
        Args:
            atom_features_dict: Dictionary of atom features from __call__
            bond_features_dict: Dictionary of bond features from __call__
            mask_prob: Probability of selecting an atom for masking
            mask_token_prob: Probability of replacing selected atom with mask token (of selected atoms)
            random_token_prob: Probability of replacing selected atom with random token (of selected atoms)
            mask_bonds: Whether to mask bonds associated with masked atoms
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (masked_atom_features, masked_bond_features, loss_masks) where:
            - masked_atom_features: Atom features with masking applied
            - masked_bond_features: Bond features with masking applied
            - loss_masks: Dict containing 'atom_mask' (1D bool tensor) and optionally 'bond_mask' (2D bool tensor)
            
        Raises:
            ValueError: If 'element' not in atom_features or if required tokenizers missing
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Validate inputs
        if 'element' not in atom_features_dict:
            raise ValueError("'element' must be in atom_features for masking")
        if 'element' not in self.atom_features:
            raise ValueError("'element' must be in self.atom_features for masking")
        
        n_atoms = len(atom_features_dict['atom_resid'])
        
        # Create copies of input dictionaries
        masked_atom_features = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() 
                            for k, v in atom_features_dict.items()}
        masked_bond_features = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() 
                            for k, v in bond_features_dict.items()}
        
        
        # Select atoms for masking
        mask_selection = torch.rand(n_atoms) < mask_prob
        selected_indices = torch.where(mask_selection)[0]
        
        if len(selected_indices) == 0:
            return masked_atom_features, masked_bond_features
        
        # Apply BERT-style masking to selected atoms
        n_selected = len(selected_indices)
        
        # Determine masking strategy for each selected atom
        mask_decisions = torch.rand(n_selected)
        mask_with_token = mask_decisions < mask_token_prob
        mask_with_random = (mask_decisions >= mask_token_prob) & (mask_decisions < mask_token_prob + random_token_prob)
        # Remaining atoms (mask_decisions >= mask_token_prob + random_token_prob) are left unchanged
        
        # Get element tokenizer
        element_tokenizer = self.tokenizers['element']
        
        # Apply masking to atom features
        for feature_name in self.atom_features:
            if feature_name not in masked_atom_features:
                continue
                
            tokenizer = self.tokenizers[feature_name]
            feature_tensor = masked_atom_features[feature_name]  # Shape: (n_atoms, 1)
            
            for i, atom_idx in enumerate(selected_indices):
                if mask_with_token[i]:
                    # Replace with mask token
                    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                        feature_tensor[atom_idx, 0] = tokenizer.mask_token_id
                elif mask_with_random[i]:
                    # Replace with random token
                    if feature_name == 'element':
                        # For element, sample from protein atoms + full_context_metals
                        vocab_items = list(element_tokenizer.get_vocab().keys())
                        # Exclude special tokens
                        vocab_items = [item for item in vocab_items if not isinstance(item, str) or not item.startswith('<')]
                        random_element = vocab_items[torch.randint(len(vocab_items), (1,)).item()]
                        feature_tensor[atom_idx, 0] = element_tokenizer.encode(random_element)
                    else:
                        # For other features, sample from valid vocabulary
                        vocab_size = tokenizer.vocab_size
                        # Exclude special tokens - assume they're at the beginning
                        start_idx = len(tokenizer.special_tokens) if hasattr(tokenizer, 'special_tokens') else 0
                        random_token = torch.randint(start_idx, vocab_size, (1,)).item()
                        feature_tensor[atom_idx, 0] = random_token
        
        # Create atom loss mask (True for atoms we want to compute loss on)
        # In BERT style, we compute loss on ALL selected atoms (masked, random, AND unchanged)
        atom_loss_mask = torch.zeros(n_atoms, dtype=torch.bool)
        atom_loss_mask[selected_indices] = True
        masked_atom_features['atom_mask'] = atom_loss_mask
        
        # Handle bond masking if requested
        if mask_bonds and bond_features_dict:
            # Only mask bond features for atoms that were actually changed (not unchanged ones)
            masked_atom_indices = selected_indices[mask_with_token | mask_with_random]
            
            # Mask bond features for masked atoms
            for feature_name in self.bond_features:
                if feature_name == 'is_bonded':
                    continue  # Skip is_bonded as specified
                    
                if feature_name not in masked_bond_features:
                    continue
                    
                tokenizer = self.tokenizers[feature_name]
                bond_matrix = masked_bond_features[feature_name]  # Shape: (n_atoms, n_atoms)
                
                if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                    # Mask rows and columns for masked atoms
                    for atom_idx in masked_atom_indices:
                        bond_matrix[atom_idx, :] = tokenizer.mask_token_id
                        bond_matrix[:, atom_idx] = tokenizer.mask_token_id
            
            # Create bond loss mask for bond_order if present
            if 'bond_order' in bond_features_dict:
                bond_loss_mask = torch.zeros((n_atoms, n_atoms), dtype=torch.bool)
                
                # Get original bonds (non-zero bond orders)
                original_bond_matrix = bond_features_dict['bond_order']
                bond_tokenizer = self.tokenizers['bond_order']
                actual_bonds = original_bond_matrix != bond_tokenizer.non_bonded_token_id
                
                # Mark masked bonds for loss computation
                for atom_idx in masked_atom_indices:
                    # Mark bonds involving this atom
                    bond_loss_mask[atom_idx, :] = actual_bonds[atom_idx, :]
                    bond_loss_mask[:, atom_idx] = actual_bonds[:, atom_idx]
                
                # Sample equal number of non-bonds for loss computation
                # Only consider upper triangle to avoid double counting
                upper_triangle = torch.triu(torch.ones(n_atoms, n_atoms, dtype=torch.bool), diagonal=1)
                
                masked_bonds_upper = bond_loss_mask & upper_triangle
                n_masked_bonds = masked_bonds_upper.sum().item()
                
                if n_masked_bonds > 0:
                    # Find non-bonds in upper triangle
                    non_bonds_upper = (~actual_bonds) & upper_triangle
                    non_bond_indices = torch.where(non_bonds_upper)
                    
                    if len(non_bond_indices[0]) >= n_masked_bonds:
                        # Sample n_masked_bonds non-bonds
                        perm = torch.randperm(len(non_bond_indices[0]))[:n_masked_bonds]
                        selected_non_bonds_i = non_bond_indices[0][perm]
                        selected_non_bonds_j = non_bond_indices[1][perm]
                        
                        # Add selected non-bonds to loss mask (both upper and lower triangle)
                        bond_loss_mask[selected_non_bonds_i, selected_non_bonds_j] = True
                        bond_loss_mask[selected_non_bonds_j, selected_non_bonds_i] = True
                    else:
                        # If not enough non-bonds, include all available non-bonds
                        bond_loss_mask[non_bonds_upper] = True
                        # Mirror to lower triangle
                        bond_loss_mask = bond_loss_mask | bond_loss_mask.T
                
                                
                    masked_bond_features['bond_mask'] = bond_loss_mask
        
        return masked_atom_features, masked_bond_features