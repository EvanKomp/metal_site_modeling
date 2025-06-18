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
from metalsitenn.placer_modules.cifutils import Chain
from metalsitenn.constants import I2E


class MetalSiteFeaturizer:
    """
    Featurizer for metal binding sites that converts Chain objects to tokenized features.
    
    Processes both atom-level and bond-level features using the configured self.tokenizers.
    Atom features are returned as (N_atoms, 1) tensors, bond features as (N_atoms, N_atoms) matrices.
    """
    tokenizers = TOKENIZERS
    
    def __init__(self, atom_features: List[str]=['element'], bond_features: List[str]=['bond_order']):
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
    
    def __call__(self, chain: Chain, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Convert a Chain object to tokenized atom and bond features.
        
        Args:
            chain: Chain object representing a metal binding site
            **kwargs: Additional arguments passed to self.tokenizers
            
        Returns:
            Tuple of (atom_features_dict, bond_features_dict, topology_features_dict)
            - atom_features_dict: Dict with keys as feature names, values as (N_atoms, 1) tensors
                                 Always includes 'atom_resid', 'atom_resname' as lists, 'positions' as (N_atoms, 3) tensor
            - bond_features_dict: Dict with keys as feature names, values as (N_atoms, N_atoms) tensors
            - topology_features_dict: Dict with ChemNet-style bond/geometry data:
                                    'bonds', 'bond_lengths', 'chirals', 'planars'
            
        Raises:
            ValueError: If bonds reference atoms not present in chain.atoms
        """
        if len(chain.atoms) == 0:
            warnings.warn("Chain has no atoms, returning empty features")
            return {}, {}, {}
        
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
        
        # Get ChemNet-style bond and geometry data
        topology_features_dict = self.get_topology(chain, atom_to_idx, **kwargs)
        
        # Process bond features
        if self.bond_features:
            bond_features_dict.update(self._process_bond_features(chain, atom_to_idx, n_atoms, **kwargs))
        
        return atom_features_dict, bond_features_dict, topology_features_dict
    
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
        atom_ishetero = []
        
        for atom_key in atom_keys:
            # atom_key is (chain_id, res_num, res_name, atom_name)
            atom_resids.append(atom_key[1])  # res_num
            atom_resnames.append(atom_key[2])  # res_name
            atom_name.append(atom_key[3])  # atom_name
            atom_ishetero.append(chain.atoms[atom_key].hetero)  # is_hetero
        
        atom_features_dict['atom_resid'] = atom_resids
        atom_features_dict['atom_resname'] = atom_resnames
        atom_features_dict['atom_name'] = atom_name
        atom_features_dict['atom_ishetero'] = atom_ishetero
        
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
            # if it doesn;t inherit from tokenizer, use float to be safe
            if issubclass(type(tokenizer), Tokenizer):
                bond_matrix = torch.full((n_atoms, n_atoms), tokenizer.non_bonded_token_id, dtype=torch.long)
            else:
                bond_matrix = torch.full((n_atoms, n_atoms), tokenizer.non_bonded_token_id, dtype=torch.float32)
            
            if feature_name == 'bond_distance':
                # Special handling for bond distance - need to compute shortest paths
                bond_matrix = self._compute_bond_distances(chain, atom_to_idx, n_atoms, tokenizer, **kwargs)
            else:
                # Process regular bond features
                for bond in chain.bonds:
                    i, j = atom_to_idx[bond.a], atom_to_idx[bond.b]
                    
                    # Extract the appropriate property based on feature name
                    if feature_name == 'bond_order':
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
    
    def get_topology(self, chain: Chain, atom_to_idx: Dict, **kwargs) -> Dict[str, torch.Tensor]:
            """
            Extract complete topology information from Chain object, matching ChemNet's topology output.
            
            Args:
                chain: Chain object
                atom_to_idx: Mapping from atom keys to indices
                **kwargs: Additional arguments (for compatibility)
                
            Returns:
                Dictionary containing:
                - 'bonds': (N_bonds, 2) tensor of atom index pairs
                - 'bond_lengths': (N_bonds,) tensor of bond lengths 
                - 'angles': (N_angles, 3) tensor of bonded angle triplets
                - 'torsions': (N_torsions, 4) tensor of bonded torsion quadruplets
                - 'chirals': (N_chirals, 4) tensor of chiral center atom indices
                - 'planars': (N_planars, 4) tensor of planar center atom indices
                - 'frames': (N_frames, 3) tensor of angle triplets for FAPE calculation
                - 'permuts': List[torch.Tensor] - automorphism permutation groups
            """
            result = {}
            
            # Extract bonds
            bond_pairs = []
            bond_lengths = []
            
            for bond in chain.bonds:
                if bond.a in atom_to_idx and bond.b in atom_to_idx:
                    i, j = atom_to_idx[bond.a], atom_to_idx[bond.b]
                    bond_pairs.append([i, j])
                    bond_lengths.append(bond.length)
            
            if bond_pairs:
                result['bonds'] = torch.tensor(bond_pairs, dtype=torch.long)
                result['bond_lengths'] = torch.tensor(bond_lengths, dtype=torch.float32)
            else:
                result['bonds'] = torch.empty((0, 2), dtype=torch.long)
                result['bond_lengths'] = torch.empty((0,), dtype=torch.float32)
            
            # Build NetworkX graph from bonds to find paths
            G = nx.Graph()
            atom_keys = list(atom_to_idx.keys())
            G.add_nodes_from(atom_keys)
            G.add_edges_from([(bond.a, bond.b) for bond in chain.bonds 
                            if bond.a in atom_to_idx and bond.b in atom_to_idx])
            
            # Find all paths of length 2 (angles: i-j-k where j is central)
            angles = self._find_all_paths_of_length_n(G, atom_to_idx, 2)
            if angles:
                result['angles'] = torch.tensor(angles, dtype=torch.long)
            else:
                result['angles'] = torch.empty((0, 3), dtype=torch.long)
            
            # Find all paths of length 3 (torsions: i-j-k-l)
            torsions = self._find_all_paths_of_length_n(G, atom_to_idx, 3)
            if torsions:
                result['torsions'] = torch.tensor(torsions, dtype=torch.long)
            else:
                result['torsions'] = torch.empty((0, 4), dtype=torch.long)
            
            # Extract chirals - (o,i,j,k) tuples where o is central atom, i,j,k are neighbors
            chiral_groups = []
            for chiral in chain.chirals:
                # Each chiral is a list of 4 atom keys: [center, neighbor1, neighbor2, neighbor3]
                if len(chiral) == 4 and all(atom_key in atom_to_idx for atom_key in chiral):
                    chiral_indices = [atom_to_idx[atom_key] for atom_key in chiral]
                    chiral_groups.append(chiral_indices)
            
            if chiral_groups:
                result['chirals'] = torch.tensor(chiral_groups, dtype=torch.long)
            else:
                result['chirals'] = torch.empty((0, 4), dtype=torch.long)
            
            # Extract planars - (o,i,j,k) tuples where o is central sp2 atom, i,j,k are neighbors
            planar_groups = []
            for planar in chain.planars:
                # Each planar is a list of 4 atom keys: [center, neighbor1, neighbor2, neighbor3] 
                if len(planar) == 4 and all(atom_key in atom_to_idx for atom_key in planar):
                    planar_indices = [atom_to_idx[atom_key] for atom_key in planar]
                    planar_groups.append(planar_indices)
            
            if planar_groups:
                result['planars'] = torch.tensor(planar_groups, dtype=torch.long)
            else:
                result['planars'] = torch.empty((0, 4), dtype=torch.long)
            
            # Extract automorphisms (permuts) - convert to atom indices and filter to observed atoms
            permuts = []
            observed_atoms = set(atom_to_idx.keys())
            
            for automorphism in chain.automorphisms:
                if isinstance(automorphism, list) and len(automorphism) > 1:
                    # Convert each automorphism group to atom indices
                    # automorphism is a list of lists, where each inner list contains equivalent atoms
                    perm_group = []
                    for equiv_group in automorphism:
                        if isinstance(equiv_group, list):
                            # Only include atoms that are in our atom_to_idx mapping
                            equiv_indices = [atom_to_idx[atom_key] for atom_key in equiv_group 
                                        if atom_key in atom_to_idx]
                            if len(equiv_indices) > 1:  # Only meaningful if >1 equivalent atoms
                                perm_group.append(equiv_indices)
                    
                    if len(perm_group) > 0:
                        # Each permutation group is a list of equivalent atom index lists
                        # Convert to tensor format matching ChemNet's expectation
                        max_group_size = max(len(group) for group in perm_group)
                        if max_group_size > 1:
                            # Pad groups to same size and create tensor
                            padded_groups = []
                            for group in perm_group:
                                padded_group = group + [-1] * (max_group_size - len(group))
                                padded_groups.append(padded_group)
                            if padded_groups:
                                permuts.append(torch.tensor(padded_groups, dtype=torch.long))
            
            result['permuts'] = permuts
            
            # Create frames for FAPE calculation (angles excluding automorphic atoms)
            # Following ChemNet approach: exclude atoms that are part of automorphisms
            skip_from_frames = set()
            for perm_tensor in permuts:
                if perm_tensor.numel() > 0:
                    # Get all atom indices from automorphism groups (excluding padding -1)
                    valid_indices = perm_tensor[perm_tensor >= 0]
                    skip_from_frames.update(valid_indices.tolist())
            
            # Filter angles to create frames, excluding automorphic atoms
            frame_angles = []
            for angle in angles:
                if not any(idx in skip_from_frames for idx in angle):
                    frame_angles.append(angle)
            
            if frame_angles:
                result['frames'] = torch.tensor(frame_angles, dtype=torch.long)
            else:
                result['frames'] = torch.empty((0, 3), dtype=torch.long)
            
            return result
    
    def _find_all_paths_of_length_n(self, G: nx.Graph, atom_to_idx: Dict, n: int) -> List[List[int]]:
            """
            Find all paths of length n in NetworkX graph and return as atom indices.
            Adapted from ChemNet's approach.
            
            Args:
                G: NetworkX graph of molecular connectivity
                atom_to_idx: Mapping from atom keys to indices  
                n: Path length (2 for angles, 3 for torsions)
                
            Returns:
                List of paths as atom index lists
            """
            def findPaths(G, u, n):
                if n == 0:
                    return [[u]]
                paths = [[u] + path for neighbor in G.neighbors(u) 
                        for path in findPaths(G, neighbor, n - 1) if u not in path]
                return paths
            
            # Find all paths of length n
            allpaths = []
            for node in G.nodes():
                paths = findPaths(G, node, n)
                for path in paths:
                    # Convert atom keys to indices
                    indices = [atom_to_idx[atom_key] for atom_key in path if atom_key in atom_to_idx]
                    if len(indices) == n + 1:  # Path of length n has n+1 nodes
                        # Canonicalize path direction (smallest index first or last)
                        if indices[0] > indices[-1]:
                            indices = indices[::-1]
                        allpaths.append(tuple(indices))
            
            # Remove duplicates and sort
            unique_paths = list(set(allpaths))
            unique_paths.sort()
            
            return [list(path) for path in unique_paths]
    
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
        # copy labels for atom features
        masked_atom_features['element_labels'] = masked_atom_features['element'].clone()
        
        
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
            assert 'bond_order' in bond_features_dict, "Bond features must include 'bond_order' for masking"
            # copy bond order labels
            masked_bond_features['bond_order_labels'] = masked_bond_features['bond_order'].clone()

            # Only mask bond features for atoms that were actually changed (not unchanged ones)
            masked_atom_indices = selected_indices[mask_with_token | mask_with_random]
            
            # Mask bond features for masked atoms
            for feature_name in self.bond_features:
                    
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
    
    def collapse_residues(self,
                        atom_features_dict: Dict[str, torch.Tensor], 
                        bond_features_dict: Dict[str, torch.Tensor],
                        collapse_rate: float = 0.3,
                        min_residues: int = 1,
                        fixed_ca: bool = True,
                        collapse_gaussian_sigma: float = 0.5,
                        center_gaussian_sigma: float = 0.2,
                        random_seed: int = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply residue collapse by selecting residues and collapsing their atoms around a center atom.
        
        For each selected residue:
        1. Choose center atom (always CA for protein residues, random for others)
        2. Optionally noise center atom position (unless fixed_ca and atom is CA)
        3. Collapse all other atoms in residue around center with Gaussian noise
        
        Args:
            atom_features_dict: Dictionary of atom features from __call__
            bond_features_dict: Dictionary of bond features from __call__
            collapse_rate: Probability of selecting a residue for collapse
            min_residues: Minimum number of residues to collapse
            fixed_ca: Whether CA atoms of protein residues remain fixed
            collapse_gaussian_sigma: Sigma for Gaussian noise when collapsing atoms
            center_gaussian_sigma: Sigma for noising center atom position (when not fixed)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (collapsed_atom_features, collapsed_bond_features) where:
            - collapsed_atom_features: Atom features with position collapse applied
            - collapsed_bond_features: Bond features (unchanged)
            - positions_labels: Original positions before collapse
            - collapse_mask: Bool tensor indicating which atoms were collapsed
            
        Raises:
            ValueError: If 'positions' not in atom_features_dict
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Validate inputs
        if 'positions' not in atom_features_dict:
            raise ValueError("'positions' must be in atom_features for collapse")
        
        # Create copies of input dictionaries
        collapsed_atom_features = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() 
                                for k, v in atom_features_dict.items()}
        collapsed_bond_features = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() 
                                for k, v in bond_features_dict.items()}
        
        # Store original positions as labels
        collapsed_atom_features['positions_labels'] = collapsed_atom_features['positions'].clone()
        
        n_atoms = len(atom_features_dict['atom_resid'])
        
        # Group atoms by residue
        residue_to_atoms = {}
        for i in range(n_atoms):
            res_key = (atom_features_dict['atom_resid'][i], 
                    atom_features_dict['atom_resname'][i])
            if res_key not in residue_to_atoms:
                residue_to_atoms[res_key] = []
            residue_to_atoms[res_key].append(i)
        
        # Select residues for collapse
        residue_keys = list(residue_to_atoms.keys())
        n_residues = len(residue_keys)
        
        # Determine number of residues to collapse
        n_collapse = max(min_residues, int(collapse_rate * n_residues))
        n_collapse = min(n_collapse, n_residues)  # Can't collapse more than available
        
        if n_collapse == 0:
            # No residues to collapse, return original with labels
            collapse_mask = torch.zeros(n_atoms, dtype=torch.bool)
            collapsed_atom_features['collapse_mask'] = collapse_mask
            return collapsed_atom_features, collapsed_bond_features
        
        # Randomly select residues to collapse
        selected_residues = torch.randperm(n_residues)[:n_collapse]
        selected_residue_keys = [residue_keys[i] for i in selected_residues]
        
        # Initialize collapse mask
        collapse_mask = torch.zeros(n_atoms, dtype=torch.bool)
        
        # Process each selected residue
        for res_key in selected_residue_keys:
            atom_indices = residue_to_atoms[res_key]
            res_name = res_key[1]  # residue name
            
            # Determine center atom
            center_atom_idx = None
            is_protein_residue = not atom_features_dict['atom_ishetero'][atom_indices[0]]
            
            if is_protein_residue:
                # For protein residues, always use CA as center
                for atom_idx in atom_indices:
                    if atom_features_dict['atom_name'][atom_idx] == 'CA':
                        center_atom_idx = atom_idx
                        break
            
            if center_atom_idx is None:
                # For non-protein residues or if CA not found, select random atom
                center_atom_idx = atom_indices[torch.randint(len(atom_indices), (1,)).item()]
            
            # Get center position
            center_pos = collapsed_atom_features['positions'][center_atom_idx].clone()
            
            # Determine if center should be noised
            is_ca_atom = (is_protein_residue and 
                        atom_features_dict['atom_name'][center_atom_idx] == 'CA')
            should_noise_center = not (fixed_ca and is_ca_atom)
            
            # Noise center position if required
            if should_noise_center:
                center_noise = torch.randn(3) * center_gaussian_sigma
                center_pos += center_noise
                collapsed_atom_features['positions'][center_atom_idx] = center_pos
            
            # Collapse all atoms in residue around center
            for atom_idx in atom_indices:
                if atom_idx == center_atom_idx:
                    # Mark center atom as collapsed if it was noised
                    if should_noise_center:
                        collapse_mask[atom_idx] = True
                else:
                    # Collapse other atoms around center with noise
                    collapse_noise = torch.randn(3) * collapse_gaussian_sigma
                    collapsed_atom_features['positions'][atom_idx] = center_pos + collapse_noise
                    collapse_mask[atom_idx] = True
        
        # Add collapse mask to features
        collapsed_atom_features['collapse_mask'] = collapse_mask
        
        return collapsed_atom_features, collapsed_bond_features