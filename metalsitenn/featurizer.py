# metalsitenn/featurizer.py
'''
* Author: Evan Komp
* Created: 6/17/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from typing import List, Dict, Tuple, Any, Union
import torch
import numpy as np
import networkx as nx
import warnings

from metalsitenn.tokenizers import TOKENIZERS, Tokenizer
from metalsitenn.placer_modules.cifutils import Chain, mutate_chain
from metalsitenn.constants import I2E, RESNAME_3LETTER

from metalsitenn.graph_data import ProteinData, BatchProteinData, make_top_k_graph

DEFAULT_FLOAT = torch.float32
DEFAULT_INT = torch.int32

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
    - custom_tokenizers_init: Dict[str, Tokenizer] - custom tokenizers for specific features instead of the default ones.
        eg. the metals that are modeled at full context can be changed in the ElementTokenizer which defaults to some set
        raises ValueError if any of the requested features are not in the default dict
    """
    tokenizers = TOKENIZERS

    def __init__(
            self, 
            atom_features: List[str] = ['element'],
            bond_features: List[str] = ['bond_order'],
            k: int = 20,
            custom_tokenizers_init: Dict[str, Tokenizer] = None
    ):
        # reasign tokenizers if custom ones are provided
        if custom_tokenizers_init is not None:
            for feature, tokenizer in custom_tokenizers_init.items():
                if feature not in self.tokenizers:
                    raise ValueError(f"Custom tokenizer for feature '{feature}' is not recognized. "
                                     f"Available features: {list(self.tokenizers.keys())}")
                self.tokenizers[feature] = tokenizer

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

    def _process_atom_features(self, chain: Chain, pdata: ProteinData, **kwargs) -> ProteinData:
        """Assign atom based features to the ProteinData object."""

        # dynamic data structure
        temp_dict = {
            'atom_name': [],
            'atom_resname': [],
            'atom_resid': [],
            'atom_ishetero': [],
        }
        for feature_name in self.atom_features:
            temp_dict[feature_name] = []

        for atom_key in chain.atoms.keys():
            # update non modelable atom features
            temp_dict['atom_name'].append(atom_key[3])
            temp_dict['atom_resname'].append(atom_key[2])
            temp_dict['atom_resid'].append(int(atom_key[1]))
            
            atom = chain.atoms[atom_key]
            temp_dict['atom_ishetero'].append(atom.hetero)

            # now add values for each feature if present
            for feature_name in self.atom_features:
                tokenizer = self.tokenizers[feature_name]
                if feature_name == 'element':
                    symbol = I2E.get(atom.element, 'X')
                    value = tokenizer.encode(symbol, **kwargs)
                elif feature_name == 'charge':
                    value = tokenizer.encode(atom.charge, **kwargs)
                elif feature_name == 'nhyd':
                    value = tokenizer.encode(atom.nhyd, **kwargs)
                elif feature_name == 'hyb':
                    value = tokenizer.encode(atom.hyb, **kwargs)
                else:
                    raise ValueError(f"Unknown atom feature: {feature_name}")
                
                temp_dict[feature_name].append(value)

        # convert to tensors and assign if possible
        for key, values in temp_dict.items():
            if not hasattr(pdata, key):
                raise ValueError(f"ProteinData does not have attribute '{key}' to assign atom features")
            # convert the featurized values to a tensor
            if key in self.atom_features:
                values = torch.tensor(values, dtype=DEFAULT_INT).unsqueeze(-1)  # [N, 1]
            elif key == 'atom_ishetero':
                values = torch.tensor(values, dtype=torch.bool).unsqueeze(-1)  # [N, 1]
            elif key == 'atom_resid':
                values = torch.tensor(values, dtype=DEFAULT_INT).unsqueeze(-1)  # [N, 1]
            else:
                values = np.array(values, dtype=str).reshape(-1, 1)  # [N, 1] - string arrays

            setattr(pdata, key, values)

        return chain, pdata
    
    def _compute_hop_distances(self, chain: Chain, atom_to_idx: Dict[Tuple, int]) -> torch.Tensor:
        """
        Compute hop distances between atoms in the chain using NetworkX.
        Returns a tensor of shape (N, N) where N is the number of atoms.
        """
        n_atoms = len(chain.atoms)
        # Initialize matrix with non-bonded tokens (distance > 7 or disconnected)
        distance_matrix = torch.full((n_atoms, n_atoms), 0, dtype=DEFAULT_INT)
        
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
                
                distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def _assign_graph(self, chain: Chain, pdata: ProteinData, atom_to_idx: Dict[Tuple, int]) -> ProteinData:
        """
        Create a graph from the chain and assign it to the ProteinData object.
        The graph is created using the make_top_k_graph function.
        """
        hop_distances = self._compute_hop_distances(chain, atom_to_idx) # [N, N]

        r = torch.tensor([atom.xyz for atom in chain.atoms.values()], dtype=DEFAULT_FLOAT)
        setattr(pdata, 'positions', r)

        src, dst, R = make_top_k_graph(
            r=r,
            hop_distances=hop_distances,
            k=self.k
        )
        edge_index = torch.stack([src, dst], dim=1)  # [E, 2]

        # also extract distances from the 2D distance matrix
        distances = R[src, dst].unsqueeze(-1)  # [E, 1]

        # assign to ProteinData
        pdata.distances = distances
        pdata.edge_index = edge_index

        return pdata
    
    def _process_bond_features(self, chain: Chain, pdata: ProteinData, atom_to_idx: Dict[Tuple, int], **kwargs) -> ProteinData:
        """Assign bond based features to the ProteinData object."""
        for bond in chain.bonds:
            if bond.a not in atom_to_idx or bond.b not in atom_to_idx:
                raise ValueError(f"Bond {bond} contains atoms not in atom_to_idx mapping")
        n_atoms = len(atom_to_idx)

        # here we will keep track of a 2D matrix at first and then use edge index to get the
        # long form tensors
        temp_dict = {}
        for feature_name in self.bond_features:
            tokenizer = self.tokenizers[feature_name]
            temp_dict[feature_name] = torch.full(
                (n_atoms, n_atoms),
                tokenizer.non_bonded_token_id,  # fill with non-bonded token id to start
                dtype=DEFAULT_INT
            )

        # process each bond for each feature
        for bond in chain.bonds:
            i, j = atom_to_idx[bond.a], atom_to_idx[bond.b]
            for feature_name in self.bond_features:
                tokenizer = self.tokenizers[feature_name]
                if feature_name == 'bond_order':
                    value = tokenizer.encode(bond.order, **kwargs)
                elif feature_name == 'is_aromatic':
                    value = tokenizer.encode(int(bond.aromatic), **kwargs)
                elif feature_name == 'is_in_ring':
                    value = tokenizer.encode(int(bond.in_ring), **kwargs)
                else:
                    raise ValueError(f"Unknown bond feature: {feature_name}")
                
                # set symmetric values
                temp_dict[feature_name][i, j] = value
                temp_dict[feature_name][j, i] = value  # symmetric

        # cut out the edge tokens from the 2D matrix
        edge_index = pdata.edge_index
        for key, values in temp_dict.items():
            if not hasattr(pdata, key):
                raise ValueError(f"ProteinData does not have attribute '{key}' to assign bond features")
            # convert the featurized values to a tensor
            values = values[edge_index[:, 0], edge_index[:, 1]].unsqueeze(-1)  # [E, 1]
            setattr(pdata, key, values)

        return pdata
    
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

    def _get_topology(self, chain: Chain, pdata: ProteinData, atom_to_idx: Dict[Tuple, int]) -> ProteinData:
        """
        Extract topology information from the chain needed for computing constrain gradients.

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

        NOTE: N_bonds != E, we will store topology in aggregated into a dict to avoid confusion with
        other attributes like edge distances. This dict is ONLY used for computing constraints and gradients.

        NOTE: Only bonds, bond lengths, planars, and chirals are currently used for gradients - the others can be used in chemnet like losses
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
        
        setattr(pdata, 'topology', result)
        return pdata
    
    def featurize_one(self, chain: Chain, **kwargs) -> ProteinData:
        """ Featurize a single Chain object into a ProteinData object.
        Args:
            chain: Chain object to featurize
            **kwargs: Additional keyword arguments for tokenizers
        Returns:
            ProteinData object with featurized data
        """
        if not isinstance(chain, Chain):
            raise ValueError("Input must be a Chain object")
        
        # Create a new ProteinData object
        pdata = ProteinData()
        
        # Assign atom features
        chain, pdata = self._process_atom_features(chain, pdata, **kwargs)
        
        # Create atom to index mapping
        atom_to_idx = {atom_key: idx for idx, atom_key in enumerate(chain.atoms.keys())}
        
        # Assign graph structure
        pdata = self._assign_graph(chain, pdata, atom_to_idx)
        
        # Assign bond features
        pdata = self._process_bond_features(chain, pdata, atom_to_idx, **kwargs)
        
        # Extract topology information
        pdata = self._get_topology(chain, pdata, atom_to_idx)
        
        return pdata
    
    def _collapse_and_noise_residues(
        self,
        pdata: ProteinData,
        resid: Union[str, int, List[Union[str, int]]],
        ca_fixed: bool=True,
        center_atom_noise_sigma: float=0.5,
        limb_atom_noise_sigma: float=0.2,
        other_atom_noise_sigma: float=None,
        time: Union[float, 'random', None] = None
    ) -> ProteinData:
        """Collapse a residue/s down to a point and noise the atoms, also assigning loss mask.

        NOTE: For amino acids, the CA atom is always the center atom
            For other residues, the atom is selected at random.
        
        Params
        ------
        - pdata: ProteinData object to modify
        - resid: Residue ID or list of IDs to collapse
        - ca_fixed: If True, and a resid is an amino acid, the CA atom will be fixed in place
        - center_atom_noise_sigma: Standard deviation for noise applied to the center atom
        - limb_atom_noise_sigma: Standard deviation for noise applied to limb atoms
        - other_atom_noise_sigma: Standard deviation for noise applied to all other atoms not targeted for collapsing. If None, no noise is applied to other atoms.
        - time: Time component for flow labels. If 'random', assigns random time. If float, uses that value. If None, no time interpolation.
        """

        # create noise mask and position labels if not already present
        if pdata.atom_noised_mask is None:
            pdata.atom_noised_mask = torch.zeros(len(pdata.positions), dtype=torch.bool)
        if pdata.position_labels is None:
            pdata.position_labels = pdata.positions.clone()
        if pdata.atom_movable_mask is None:
            pdata.atom_movable_mask = torch.zeros(len(pdata.positions), dtype=torch.bool)

        # for resid as a list
        if not isinstance(resid, list):
            resid = [resid]

        # Track all atoms that are targeted for collapsing
        all_collapsed_atoms_mask = torch.zeros(len(pdata.positions), dtype=torch.bool)

        for resid_ in resid:
            # mark atoms in this residue
            resid = int(resid_)
            is_resid_mask = (pdata.atom_resid == resid_).squeeze(-1)
            # assign the mask for noised atoms
            pdata.atom_noised_mask[is_resid_mask] = True
            all_collapsed_atoms_mask[is_resid_mask] = True

            # find the center atom
            # first get the resname for this residue
            resname = pdata.atom_resname[torch.where(is_resid_mask)[0][0]]
            is_aa = False
            if resname in RESNAME_3LETTER:
                is_aa = True
                # atom name is a list of str so we can't do tensor masking
                center_atom_idx = np.where(((pdata.atom_name == 'CA').flatten() * is_resid_mask.numpy().flatten()).astype(bool))[0][0]
            else:
                # for non-amino acids, select a random atom as the center
                center_atom_idx = np.random.choice(np.where(is_resid_mask)[0])

            # determine center atom position
            if ca_fixed and is_aa:
                # keep the CA atom fixed
                center_atom_position = pdata.positions[center_atom_idx].clone()
                # turn the mask off for the center atom
                pdata.atom_noised_mask[center_atom_idx] = False
            else:
                # noise the center atom position
                center_atom_position = pdata.positions[center_atom_idx] + \
                    torch.randn(3, dtype=DEFAULT_FLOAT) * center_atom_noise_sigma
            # shift all atoms to equal the center atom position
            pdata.positions[is_resid_mask] = center_atom_position
            # generate noise around the center atom
            is_resid_but_not_center = is_resid_mask.clone()
            is_resid_but_not_center[center_atom_idx] = False
            # noise the limb atoms
            pdata.positions[is_resid_but_not_center] += torch.randn(
                is_resid_but_not_center.sum(), 3, dtype=DEFAULT_FLOAT) * limb_atom_noise_sigma

        # Handle other atoms noise if specified
        if other_atom_noise_sigma is not None:
            # Find all atoms NOT targeted for collapsing
            other_atoms_mask = ~all_collapsed_atoms_mask
            
            # If ca_fixed is True, respect that for amino acid CA atoms in other residues too
            if ca_fixed:
                # Find all CA atoms in amino acids that are not being collapsed
                for i, (atom_name, resname) in enumerate(zip(pdata.atom_name.flatten(), pdata.atom_resname.flatten())):
                    if (atom_name == 'CA' and resname in RESNAME_3LETTER and 
                        other_atoms_mask[i]):
                        other_atoms_mask[i] = False  # Don't noise CA atoms if ca_fixed=True
            
            # Apply noise to other atoms
            if other_atoms_mask.any():
                pdata.positions[other_atoms_mask] += torch.randn(
                    other_atoms_mask.sum(), 3, dtype=DEFAULT_FLOAT) * other_atom_noise_sigma
                # Mark these atoms as noised
                pdata.atom_noised_mask[other_atoms_mask] = True

        # update all noised atoms to be movable
        pdata.atom_movable_mask[pdata.atom_noised_mask] = True

        # assign the target flow vectors
        # set the flow labels to be the difference between the original and noised positions
        pdata.position_flow_labels = pdata.position_labels - pdata.positions

        # if we have a time component specified, assign it and update the actual positions
        if time is not None:
            if isinstance(time, str) and time == 'random':
                # assign a random time value
                time = torch.rand(1, dtype=DEFAULT_FLOAT).item()
            elif isinstance(time, (int, float)):
                time = float(time)
            else:
                raise ValueError(f"Invalid time value: {time}. Must be 'random', int, or float.")
            
            # update the currently fully noised position labels with an interpolation at the specified time
            pdata.positions = pdata.positions + pdata.position_flow_labels * time

            # assign the time component if not already present
            if pdata.time is None:
                pdata.time = torch.tensor([time], dtype=DEFAULT_FLOAT)

        # if we are in here, we should invalidate the distances attribute
        pdata.distances = None

        return pdata

    def _mask_atoms(
        self, 
        pdata: ProteinData, 
        indices_to_mask: List[int],
        indices_to_tweak: List[int] = None,
        indices_to_keep: List[int] = None
    ) -> ProteinData:
        """Mask specific atoms in the ProteinData object with BERT-style strategy.
        
        NOTE: The element is of course masked, but information that specified the atom will also be masked.
        Eg. any atom features are set to mask, bonds associated with the atom are removed including aromatic and ring indicators.

        NOTE: This method could potentially delete topology information associated with masked atoms to prevent leakage. For now,
        instead we will manully delete L1 gradient features for masked atoms. This also allows us to easily keep
        eg. planar constraints for one atom that is not masked in a constraint that contains a masked atom.

        Args:
            pdata: ProteinData object to modify
            indices_to_mask: List of atom indices to mask with [MASK] token
            indices_to_tweak: List of atom indices to replace with random tokens (BERT 10% random)
            indices_to_keep: List of atom indices to keep unchanged (BERT 10% unchanged)
        Returns:
            Modified ProteinData object with masked atoms
        """
        
        # set element labels
        if pdata.element_labels is None:
            pdata.element_labels = pdata.element.clone()

        # Combine all indices that are being modified for atom_masked_mask
        all_modified_indices = set()
        if indices_to_mask:
            all_modified_indices.update(indices_to_mask)
        if indices_to_tweak:
            all_modified_indices.update(indices_to_tweak)
        if indices_to_keep:
            all_modified_indices.update(indices_to_keep)
        
        # Set atom_masked_mask for all modified atoms
        if all_modified_indices:
            if pdata.atom_masked_mask is None:
                pdata.atom_masked_mask = torch.zeros(len(pdata.element), dtype=torch.bool)
            for idx in all_modified_indices:
                pdata.atom_masked_mask[idx] = True

        # Process atom features with BERT-style masking
        for feature_name in self.atom_features:
            assert getattr(pdata, feature_name) is not None, f"Feature {feature_name} not found in ProteinData"
            feature_tensor = getattr(pdata, feature_name)
            tokenizer = self.tokenizers[feature_name]
            
            # Mask tokens (80% in BERT)
            if indices_to_mask:
                mask_id = tokenizer.mask_token_id
                feature_tensor[indices_to_mask] = mask_id
            
            # Random tokens (10% in BERT)
            if indices_to_tweak:
                for idx in indices_to_tweak:
                    # Get random token from vocabulary (excluding special tokens)
                    vocab_size = tokenizer.vocab_size
                    # Exclude mask, unk tokens if they exist
                    exclude_tokens = []
                    if tokenizer.mask_token_id is not None:
                        exclude_tokens.append(tokenizer.mask_token_id)
                    if tokenizer.unk_token_id is not None:
                        exclude_tokens.append(tokenizer.unk_token_id)
                    
                    valid_tokens = [i for i in range(vocab_size) if i not in exclude_tokens]
                    random_token = torch.randint(0, len(valid_tokens), (1,)).item()
                    feature_tensor[idx] = valid_tokens[random_token]
            
            # Keep original tokens (10% in BERT) - no modification needed for indices_to_keep

        # Get all edges associated with a MASKED atom eg. not the modified or unmodified ones
        # find all edges in the edge index that contain oned of the indices_to_mask
        left_has_masked_atom = torch.isin(pdata.edge_index[:, 0], torch.Tensor(indices_to_mask))
        right_has_masked_atom = torch.isin(pdata.edge_index[:, 1], torch.Tensor(indices_to_mask))
        # Combine both masks to find edges with masked atoms
        masked_edges_mask = left_has_masked_atom | right_has_masked_atom
        # Mask all edge features associated with masked atoms
        for feature_name in self.bond_features:
            if hasattr(pdata, feature_name) and getattr(pdata, feature_name) is not None:
                feature_tensor = getattr(pdata, feature_name)
                mask_id = self.tokenizers[feature_name].mask_token_id
                # Apply mask to all edges that have masked atoms
                feature_tensor[masked_edges_mask] = mask_id

        return pdata
    
    def _anonymize_metals_for_classification(self, pdata: ProteinData, include_special_tokens: bool = True) -> ProteinData:
        """
        Anonymize metals for global classification task by converting all metals to <METAL> token
        and masking associated features. Sets global label vector for metal counts.
        
        This method:
        1. Determines metal count labels from current element tokens
        2. Converts all metal elements to <METAL> token
        3. Masks all node and edge features associated with metal atoms
        4. Assigns global labels to pdata.global_labels
        
        Args:
            pdata: ProteinData object to modify
            include_special_tokens: If True, include special tokens in label vector
            
        Returns:
            Modified ProteinData object with anonymized metals and global labels
        """
        # Get element tokenizer
        element_tokenizer = self.tokenizers['element']
        
        # Use cached metal token IDs for efficient vectorized detection
        metal_token_ids = element_tokenizer.metal_token_ids
        
        # Find metal atoms using isin (much faster than loop)
        metal_mask = torch.isin(pdata.element, metal_token_ids)
        
        # Extract metal tokens for count vector calculation
        metal_tokens = pdata.element[metal_mask]
        
        # Generate metal count labels using ElementTokenizer method
        metal_counts = element_tokenizer.encode_metal_composition_counts_from_tokens(
            metal_tokens, include_special_tokens=include_special_tokens
        )
        
        # Store as global labels
        # make sure default float
        pdata.global_labels = metal_counts.to(dtype=DEFAULT_FLOAT)
        
        # Convert all metals to <METAL> token (vectorized)
        metal_token_id = element_tokenizer.metal_token_id
        pdata.element[metal_mask] = metal_token_id
        
        # Mask all node features for metal atoms (vectorized)
        for feature_name in self.atom_features:
            # skip element since we already converted to metal token
            if feature_name == 'element':
                continue
            if hasattr(pdata, feature_name) and getattr(pdata, feature_name) is not None:
                feature_tensor = getattr(pdata, feature_name)
                mask_id = self.tokenizers[feature_name].mask_token_id
                feature_tensor[metal_mask] = mask_id
        
        # Mask all edge features associated with metal atoms (vectorized)
        if hasattr(pdata, 'edge_index') and pdata.edge_index is not None:
            edge_index = pdata.edge_index
            # Vectorized edge detection - check if either endpoint is a metal
            metal_edges_mask = metal_mask[edge_index[:, 0]] | metal_mask[edge_index[:, 1]]
            
            # Mask bond features for these edges
            for feature_name in self.bond_features:
                if hasattr(pdata, feature_name) and getattr(pdata, feature_name) is not None:
                    feature_tensor = getattr(pdata, feature_name)
                    mask_id = self.tokenizers[feature_name].mask_token_id
                    feature_tensor[metal_edges_mask] = mask_id
        
        return pdata

    def __call__(
        self,
        chains: Union[List[Chain], Chain],
        # for node classification task
        node_mlm_do: bool = False,
        node_mlm_rate: float = 0.15,
        node_mlm_subrate_tweak: float = 0.1,
        node_mlm_subrate_keep: float = 0.1,
        # for global classification task
        metal_classification: bool = False,
        metal_classification_include_special_tokens: bool = True,
        # for residue collapsing / denoising
        residue_collapse_do: bool = False,
        residue_collapse_specific_residues: Union[List[Union[str, int]], str, int] = None,
        residue_collapse_rate: float = 0.3,
        residue_collapse_ca_fixed: bool = False,
        residue_collapse_center_atom_noise_sigma: float = 0.5,
        residue_collapse_limb_atom_noise_sigma: float = 0.2,
        residue_collapse_other_atom_noise_sigma: float = None,
        residue_collapse_time: Union[float, 'random', None] = None,  # 'random' or a float value
        movable_atoms: str = 'noised', # 'none', 'all', 'noised', 'noised_and_side_chains'
        # custom behavior eg. mutating
        mutations: Tuple[int, str] = None,  # (resid, new_resname) to mutate a residue
        **kwargs
    ):
        """
        Featurize Chains into ProteinData objects.

        Optional functionalities:
        - Node masking for MLM training tasks
        - Metal anonymization for global classification tasks
        - Residue collapsing for denoising tasks
        
        Args:
            chains: List of Chain objects or a single Chain to featurize
            node_mlm_do: If True, apply MLM-style masking to nodes
            node_mlm_rate: Probability of masking nodes for MLM training (0.0 to 1.0)
            node_mlm_subrate_tweak: Probability of replacing masked nodes with random tokens (default 0.1)
            node_mlm_subrate_keep: Probability of keeping original tokens (default 0.1)
            metal_classification: If True, anonymize metals for global classification
            metal_classification_include_special_tokens: Include special tokens in metal count vector
            residue_collapse_do: If True, collapse specified residues and apply noising
            residue_collapse_specific_residues: Residue IDs to collapse (list of str/int or single str/int)
            residue_collapse_rate: Probability of collapsing residues (0.0 to 1.0)
            residue_collapse_ca_fixed: If True, keep CA atom fixed during collapse / noising
            residue_collapse_center_atom_noise_sigma: Standard deviation for center atom noise
            residue_collapse_limb_atom_noise_sigma: Standard deviation for limb atom noise
            residue_collapse_other_atom_noise_sigma: Standard deviation for other atom noise
            residue_collapse_time: Time component for flow labels ('random' or float)
            movable_atoms: Specification of movable atoms during noising
                - 'none': No atoms are movable
                - 'all': All atoms are movable
                - 'noised': Only noised atoms are movable eg. exactly the noise mask
                - 'noised_and_side_chains': Any side chain atom is movable in addition to noised atoms
            mutations: Tuple of (resid, new_resname) to mutate a residue
            **kwargs: Additional keyword arguments for customization
        Returns:
            BatchProteinData object containing featurized data
        """
        if isinstance(chains, Chain):
            chains = [chains]

        if not isinstance(chains, list):
            raise ValueError("Input must be a Chain object or a list of Chain objects")
        
        # Initialize list to hold featurized ProteinData objects
        featurized_data = []

        for chain in chains:
            features = self.featurize_one(chain, **kwargs)

            ########################################
            # COLLAPSING, NOISING, AND MUTATIONS
            ########################################
            if residue_collapse_do:
                resids_to_collapse = []
                # select residue ids at random
                if residue_collapse_rate > 0.0:
                    all_resids = features.atom_resid.unique()
                    n_residues = len(all_resids)
                    n_to_collapse = round(n_residues * residue_collapse_rate)
                    if n_to_collapse > 0:
                        # select random residues to collapse
                        selected_resids = all_resids[torch.randperm(n_residues)[:n_to_collapse]]
                        resids_to_collapse.extend(selected_resids.tolist())

                # if specific residues are provided, add them to the list
                if residue_collapse_specific_residues is not None:
                    if not isinstance(residue_collapse_specific_residues, list):
                        residue_collapse_specific_residues = [residue_collapse_specific_residues]
                    resids_to_collapse.extend(residue_collapse_specific_residues)

                # remove duplicates and convert to int
                resids_to_collapse = list(set(resids_to_collapse))
                resids_to_collapse = [int(resid) for resid in resids_to_collapse]

                # do it
                features = self._collapse_and_noise_residues(
                    features,
                    resid=resids_to_collapse,
                    ca_fixed=residue_collapse_ca_fixed,
                    center_atom_noise_sigma=residue_collapse_center_atom_noise_sigma,
                    limb_atom_noise_sigma=residue_collapse_limb_atom_noise_sigma,
                    other_atom_noise_sigma=residue_collapse_other_atom_noise_sigma,
                    time=residue_collapse_time
                )

            # apply mutations
            if mutations is not None:
                raise NotImplementedError("Mutations are not yet implemented in the featurizer.")

            # assign the atom movable mask based on the specified behavior\
            if features.atom_movable_mask is None:
                # this might be true if not collapsing residues be still should do it outside of the
                # collapsing logic incase mutations were made
                features.atom_movable_mask = torch.zeros(len(features.positions), dtype=torch.bool)

            if movable_atoms == 'none':
                features.atom_movable_mask = torch.zeros(len(features.positions), dtype=torch.bool)
            elif movable_atoms == 'all':
                features.atom_movable_mask = torch.ones(len(features.positions), dtype=torch.bool)
            elif movable_atoms == 'noised':
                # already set in the _collapse_and_noise_residues method or as zeros above if not
                pass
            elif movable_atoms == 'noised_and_side_chains':
                # noised already set, now we just have to find all atoms in 
                # amino acids that are not CA atoms and set them as movable
                is_aa_mask = features.atom_resname.isin(RESNAME_3LETTER)
                is_ca_mask = (features.atom_name == 'CA').flatten()
                to_move_mask = is_aa_mask & ~is_ca_mask
                features.atom_movable_mask[to_move_mask] = True
            else:
                raise ValueError(f"Invalid movable_atoms value: {movable_atoms}. "
                                    "Must be 'none', 'all', 'noised', or 'noised_and_side_chains'.")
            
            ########################################
            # NODE MASKING FOR MLM TASKS
            ########################################
            if node_mlm_do:
                # Determine number of nodes to mask based on rate
                n_nodes = len(features.element)
                n_to_mask = round(n_nodes * node_mlm_rate)
                
                if n_to_mask > 0:
                    # Randomly select nodes to mask
                    indices_to_mask = set(torch.randperm(n_nodes)[:n_to_mask].tolist())
                    
                    # Determine subrates for tweaking and keeping
                    n_to_tweak = round(n_to_mask * node_mlm_subrate_tweak)
                    n_to_keep = round(n_to_mask * node_mlm_subrate_keep)
                    
                    # select indices to tweak and keep from the already masked indices
                    masked_indices_random_perm = torch.randperm(indices_to_mask)
                    indices_to_tweak = set(masked_indices_random_perm[:n_to_tweak].tolist())
                    indices_to_keep = set(masked_indices_random_perm[n_to_tweak:n_to_tweak + n_to_keep].tolist())
                    # remove indices from mask that are in tweak or keep
                    indices_to_mask -= indices_to_tweak
                    indices_to_mask -= indices_to_keep

                # Apply BERT-style masking
                # do this outside of the if statement so that even if no nodes are masked,
                # we still call the method to ensure other attributes are assigned correctly.
                features = self._mask_atoms(
                    features,
                    indices_to_mask=list(indices_to_mask),
                    indices_to_tweak=list(indices_to_tweak),
                    indices_to_keep=list(indices_to_keep)
                )

            ########################################
            # METAL ANONYMIZATION FOR CLASSIFICATION TASKS
            ########################################
            if metal_classification:
                features = self._anonymize_metals_for_classification(
                    features,
                    include_special_tokens=metal_classification_include_special_tokens
                )
            # Append the featurized ProteinData object to the list
            featurized_data.append(features)

        # reset the distances after we maybe screwed with them
        if self.distances is None:
            features.set_distances()

        # convert to BatchProteinData
        batch_data = BatchProteinData.from_list(featurized_data)

        return batch_data
                




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


    

            

                
        

    
