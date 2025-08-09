# metalsitenn/graph_data.py
'''
* Author: Evan Komp
* Created: 8/5/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from dataclasses import dataclass, fields

from typing import List, Tuple, Dict, Any, Optional
import torch
import numpy as np
import copy


def make_top_k_graph(r, hop_distances, k=10):
    """
    Create a top-k graph based on bond distances.
    
    Args:
        r (torch.Tensor): Positions of atoms, shape (N, 3).
        hop_distances (torch.Tensor): Hop distances between each atom.
        k (int): Number of nearest neighbors to consider for each atom.
            Up to half are determined by bonding patterns, 
            the rest by distance.
    """
    N = r.shape[0]
    _,idx = torch.topk(hop_distances.masked_fill(hop_distances==0,999), min(k//2+1,N), largest=False)
    distance_mask = torch.zeros_like(hop_distances,dtype=bool).scatter_(1,idx,True)
    distance_mask = distance_mask & (hop_distances>0)

    # then pull from actual angstrom distances
    # first compute pairwise distances|
    R = torch.cdist(r, r)  # (N, N)
    # fill in distance with the ones we have already chosen so that they are insta chosen
    R_ = R.masked_fill(distance_mask, 0.0)
    _,idx = torch.topk(R_, min(k+1,N), largest=False)
    r_mask = torch.zeros_like(R_, dtype=bool).scatter_(1, idx, True)

    # get edges
    src,dst = torch.where(r_mask.fill_diagonal_(False)) # self edge deleted
    return src, dst, R

# mutable
@dataclass(frozen=False)
class ProteinData:
    """
    Data class for protein atomic data.
    N = num atoms
    E = num edges
    B = num systems in batch N

    """
    # atom info
    element: torch.Tensor=None  # [N, 1]
    charge: torch.Tensor=None  # [N, 1]
    nhyd: torch.Tensor=None  # [N, 1]
    hyb: torch.Tensor=None  # [N, 1]
    positions: torch.Tensor=None  # [N, 3]
    atom_movable_mask: torch.Tensor=None  # [N, 1] - mask for atoms that can be moved. This should contain at least all indices in atom_noised_mask

    # for posterity
    atom_name: np.ndarray[str]=None  # [N, 1] - atom names
    atom_resname: np.ndarray[str]=None  # [N, 1] - residue names
    atom_resid: torch.Tensor=None  # [N, 1] - residue ids
    atom_ishetero: torch.Tensor=None  # [N, 1] - is hetero atom

    # edge info
    distances: torch.Tensor=None  # [E, 1] - distances between atoms
    bond_order: torch.Tensor=None # [E, 1]
    is_aromatic: torch.Tensor=None  # [E, 1]
    is_in_ring: torch.Tensor=None  # [E, 1]
    edge_index: torch.Tensor=None  # [E, 2]

    # topology
    topology: Dict[str, torch.Tensor]=None  
    # Includes
    # bonds [N_bonds, 2] - pairs of atom indexes that are bonded
    # bond_lengths [N_bonds, 1] - equilibrium bond lengths of the bonds
    # planars [O,4]
    # chirals [O,4] O here is number of contraints, and 5 is the number if indexes required to specify the comstraint

    # global features
    global_features: torch.Tensor=None  # [1, d]
    time: torch.Tensor=None  # [1, 1] - time of the system, if applicable
    pdb_id: np.ndarray[str]=None  # [1, 1] - PDB identifier

    # attributes related to collating / loss calculation
    atom_masked_mask: torch.Tensor=None  # [N, 1] - mask for atoms that were masked
    element_labels: torch.Tensor=None  # [N, 1] - labels
    element_loss_weights: torch.Tensor=None # [N, 1] - per atom weights for element loss

    global_labels: torch.Tensor=None  # [1, d] - labels for global tasks
    global_loss_weights: torch.Tensor=None  # [1, d] - per example weights for global tasks

    atom_noised_mask: torch.Tensor=None  # [N, 1] - mask for atoms that were noised
    position_flow_labels: torch.Tensor=None  # [N, 3] - labels for positions in flow tasks
    position_labels: torch.Tensor=None  # [N, 3] - labels for positions
    position_loss_weights: torch.Tensor=None  # [N, 3] - per atom weights for denoising loss


    def __repr__(self) -> str:
        """
        Return shapes or lengths of data structures in a readable format.
        """
        repr_str = "ProteinData(\n"
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                repr_str += f"  {field}=None,\n"
            elif hasattr(value, 'shape'):
                repr_str += f"  {field}: shape={tuple(value.shape)},\n"
            elif isinstance(value, list):
                repr_str += f"  {field}: len={len(value)},\n"
            else:
                repr_str += f"  {field}={value},\n"
        repr_str += ")"
        return repr_str
    
    def set_distances(self):
        """
        Calculate and set distances based on current positions.
        
        This method computes pairwise distances between atoms and sets the
        `distances` attribute.
        """
        if self.positions is None or self.edge_index is None:
            raise ValueError("Positions must be set before calculating distances.")
        
        # Compute pairwise distances
        R = torch.cdist(self.positions, self.positions)
        # Fill diagonal with zeros (self-distances)
        R.fill_diagonal_(0.0)
        # Store distances in the ProteinData object using edge index
        src, dst = self.edge_index.t()
        distances = R[src, dst].unsqueeze(-1)  # [E, 1]
        self.distances = distances


    def clone(self) -> 'ProteinData':
        """
        Create a deep copy of the ProteinData instance.
        
        Returns:
            ProteinData: A new instance with the same data.
        """
        data = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                data[field] = value.clone()
            elif isinstance(value, np.ndarray):
                data[field] = value.copy()
            else:
                data[field] = copy.deepcopy(value)
        return ProteinData(**data)
    
    def save(self, path: str) -> None:
        """
        Save ProteinData to file using PyTorch serialization.
        
        Args:
            path: File path to save to (e.g., 'protein_data.pt')
        """
        state_dict = {}
        
        for field in fields(self):
            value = getattr(self, field.name)
            state_dict[field.name] = value
        
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'ProteinData':
        """
        Load ProteinData from file.
        
        Args:
            path: File path to load from
            device: Optional device to load tensors to
            
        Returns:
            ProteinData instance
        """
        state_dict = torch.load(path, map_location=device)
        kwargs = {}
        
        for field in fields(cls):
            field_name = field.name
            
            if field_name in state_dict:
                kwargs[field_name] = state_dict[field_name]
            else:
                # Handle missing fields with defaults
                kwargs[field_name] = field.default if field.default != dataclass.MISSING else None
        
        instance = cls(**kwargs)
        
        # Move tensors to device if specified
        if device is not None:
            instance = instance.to(device)
            
        return instance


class BatchProteinData:
    """
    Batch container for multiple ProteinData instances.
    Handles vertical stacking with proper index shifting for edge_index and topology.
    """
    
    def __init__(self, protein_data_list: List[ProteinData]):
        """
        Create batch from list of ProteinData instances.
        
        Args:
            protein_data_list: List of ProteinData instances to batch
        """
        if not protein_data_list:
            raise ValueError("Cannot create batch from empty list")
        
        self.batch_size = len(protein_data_list)
        self._atom_counts = [p.positions.shape[0] if p.positions is not None else 0 
                            for p in protein_data_list]
        self._atom_offsets = [0] + list(torch.cumsum(torch.tensor(self._atom_counts[:-1]), dim=0).tolist())
        
        # Stack all tensor attributes
        self._stack_attributes(protein_data_list)
    
    def _stack_attributes(self, protein_data_list: List[ProteinData]):
        """Stack all attributes from protein data list with proper index shifting."""
        # Initialize all attributes to None
        for field in fields(ProteinData):
            setattr(self, field.name, None)
        
        # Create batch tensor for tracking which atoms belong to which sample
        total_atoms = sum(self._atom_counts)
        batch_tensor = torch.cat([torch.full((count,), i, dtype=torch.long) 
                                 for i, count in enumerate(self._atom_counts)])
        self.batch = batch_tensor
        
        # Stack tensor attributes
        self._stack_tensor_attributes(protein_data_list)
        
        # Stack numpy attributes  
        self._stack_numpy_attributes(protein_data_list)
        
        # Handle topology with index shifting
        self._stack_topology(protein_data_list)
    
    def _stack_tensor_attributes(self, protein_data_list: List[ProteinData]):
        """Stack regular tensor attributes (non-topology, non-numpy)."""
        tensor_fields = ['element', 'charge', 'nhyd', 'hyb', 'positions', 'atom_movable_mask',
                        'atom_resid', 'atom_ishetero', 'distances', 'bond_order', 'is_aromatic',
                        'is_in_ring', 'atom_masked_mask', 'element_labels', 'element_loss_weights',
                        'atom_noised_mask', 'position_flow_labels', 'position_labels', 
                        'position_loss_weights']
        
        for field_name in tensor_fields:
            tensors = [getattr(p, field_name) for p in protein_data_list 
                      if getattr(p, field_name) is not None]
            if tensors:
                setattr(self, field_name, torch.cat(tensors, dim=0))
        
        # Handle edge_index separately (needs index shifting)
        edge_indices = []
        for i, p in enumerate(protein_data_list):
            if p.edge_index is not None:
                shifted_edge_index = p.edge_index + self._atom_offsets[i]
                edge_indices.append(shifted_edge_index)
        if edge_indices:
            self.edge_index = torch.cat(edge_indices, dim=0)
        
        # Handle global attributes (no stacking needed for per-sample attributes)
        global_fields = ['global_features', 'time', 'global_labels', 'global_loss_weights']
        for field_name in global_fields:
            tensors = [getattr(p, field_name) for p in protein_data_list 
                      if getattr(p, field_name) is not None]
            if tensors:
                setattr(self, field_name, torch.cat(tensors, dim=0))
    
    def _stack_numpy_attributes(self, protein_data_list: List[ProteinData]):
        """Stack numpy array attributes."""
        numpy_fields = ['atom_name', 'atom_resname', 'pdb_id']
        
        for field_name in numpy_fields:
            arrays = [getattr(p, field_name) for p in protein_data_list 
                     if getattr(p, field_name) is not None]
            if arrays:
                setattr(self, field_name, np.concatenate(arrays, axis=0))
    
    def _stack_topology(self, protein_data_list: List[ProteinData]):
        """Stack topology dictionaries with proper index shifting."""
        topologies = [p.topology for p in protein_data_list if p.topology is not None]
        if not topologies:
            return
        
        # Get all unique keys across all topologies
        all_keys = set()
        for topo in topologies:
            all_keys.update(topo.keys())
        
        stacked_topology = {}
        
        for key in all_keys:
            tensors = []
            for i, p in enumerate(protein_data_list):
                if p.topology is not None and key in p.topology:
                    tensor = p.topology[key]
                    if key == 'permuts':
                        new_permuts = []
                        for permut in tensor:
                            # Shift indices in permuts by the atom offset
                            shifted_permut = permut + self._atom_offsets[i]
                            new_permuts.append(shifted_permut)
                        tensors.extend(new_permuts)
                    # Shift indices for all keys except bond_lengths
                    elif key != 'bond_lengths' and tensor.dtype in [torch.long, torch.int32, torch.int64]:
                        tensor = tensor + self._atom_offsets[i]
                        tensors.append(tensor)
                    else:
                        tensors.append(tensor)
            
            if tensors:
                if key == 'permuts':
                    # done, already a list of tensors
                    stacked_topology[key] = tensors
                else:
                    # stack tensors along the first dimension
                    stacked_topology[key] = torch.cat(tensors, dim=0)
        
        self.topology = stacked_topology if stacked_topology else None
    
    def get(self, idx: int) -> ProteinData:
        """
        Get a single ProteinData instance from the batch.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            ProteinData instance
        """
        if idx < 0 or idx >= self.batch_size:
            raise IndexError(f"Index {idx} out of range for batch size {self.batch_size}")
        
        return self.to_protein_data_list()[idx]
    
    def __getitem__(self, idx: int) -> ProteinData:
        """
        Get a single ProteinData instance using indexing.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            ProteinData instance
        """
        return self.get(idx)
    
    def save(self, path: str) -> None:
        """
        Save BatchProteinData to file.
        
        Args:
            path: File path to save to
        """
        state_dict = {
            'batch_size': self.batch_size,
            '_atom_counts': self._atom_counts,
            '_atom_offsets': self._atom_offsets,
        }
        
        # Save all attributes
        for field in fields(ProteinData):
            attr = getattr(self, field.name, None)
            if attr is not None:
                state_dict[field.name] = attr
        
        # Save batch tensor
        state_dict['batch'] = self.batch
        
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'BatchProteinData':
        """
        Load BatchProteinData from file.
        
        Args:
            path: File path to load from
            device: Optional device to load tensors to
            
        Returns:
            BatchProteinData instance
        """
        state_dict = torch.load(path, map_location=device)
        
        # Create empty instance
        instance = cls.__new__(cls)
        
        # Restore metadata
        instance.batch_size = state_dict['batch_size']
        instance._atom_counts = state_dict['_atom_counts']
        instance._atom_offsets = state_dict['_atom_offsets']
        
        # Restore attributes
        for field in fields(ProteinData):
            field_name = field.name
            if field_name in state_dict:
                setattr(instance, field_name, state_dict[field_name])
            else:
                setattr(instance, field_name, None)
        
        # Restore batch tensor
        instance.batch = state_dict['batch']
        
        return instance
    
    def to_protein_data_list(self) -> List[ProteinData]:
        """
        Convert batch back to list of individual ProteinData instances.
        
        Returns:
            List of ProteinData instances
        """
        protein_data_list = []
        
        for i in range(self.batch_size):
            # Get atom mask for this sample
            atom_mask = (self.batch == i)
            atom_indices = torch.where(atom_mask)[0]
            
            # Extract atom-level data
            data = {}
            atom_fields = ['element', 'charge', 'nhyd', 'hyb', 'positions', 'atom_movable_mask',
                        'atom_resid', 'atom_ishetero', 'atom_masked_mask', 'element_labels', 
                        'element_loss_weights', 'atom_noised_mask', 'position_flow_labels',
                        'position_labels', 'position_loss_weights']
            
            for field_name in atom_fields:
                attr = getattr(self, field_name, None)
                if attr is not None:
                    data[field_name] = attr[atom_mask]
            
            # Handle atom-level numpy arrays (NOT pdb_id)
            numpy_fields = ['atom_name', 'atom_resname']
            for field_name in numpy_fields:
                attr = getattr(self, field_name, None)
                if attr is not None:
                    data[field_name] = attr[atom_mask.cpu().numpy()]
            
            # Handle global numpy arrays (like pdb_id)
            global_numpy_fields = ['pdb_id']
            for field_name in global_numpy_fields:
                attr = getattr(self, field_name, None)
                if attr is not None:
                    data[field_name] = attr[i:i+1]  # Keep as [1,1] array
            
            # Handle edges
            if self.edge_index is not None:
                # Find edges where both nodes belong to this sample
                src, dst = self.edge_index.t()
                edge_mask = atom_mask[src] & atom_mask[dst]
                
                if edge_mask.any():
                    sample_edge_index = self.edge_index[edge_mask] - self._atom_offsets[i]
                    data['edge_index'] = sample_edge_index
                    
                    # Extract corresponding edge features
                    edge_fields = ['distances', 'bond_order', 'is_aromatic', 'is_in_ring']
                    for field_name in edge_fields:
                        attr = getattr(self, field_name, None)
                        if attr is not None:
                            data[field_name] = attr[edge_mask]
            
            # Handle topology
            if self.topology is not None:
                sample_topology = {}
                for key, value in self.topology.items():
                    if key == 'permuts':
                        # Handle permuts - list of tensors
                        sample_permuts = []
                        for permut_tensor in value:
                            # Check if all indices in this permut belong to current sample
                            valid_mask = atom_mask[permut_tensor.flatten()]
                            if valid_mask.all():
                                # Shift indices and add to sample
                                shifted_permut = permut_tensor - self._atom_offsets[i]
                                sample_permuts.append(shifted_permut)
                        if sample_permuts:
                            sample_topology[key] = sample_permuts
                    elif key == 'bond_lengths':
                        # Bond lengths don't contain indices - filter by relevant bonds
                        if 'bonds' in self.topology:
                            bond_indices = self.topology['bonds']
                            bond_mask = atom_mask[bond_indices[:, 0]] & atom_mask[bond_indices[:, 1]]
                            if bond_mask.any():
                                sample_topology[key] = value[bond_mask]
                    else:
                        # Regular tensor with indices - filter and shift
                        if isinstance(value, torch.Tensor) and value.dtype in [torch.long, torch.int32, torch.int64]:
                            # Find constraints where all referenced atoms are in this sample
                            constraint_mask = torch.all(atom_mask[value], dim=1)
                            if constraint_mask.any():
                                sample_topology[key] = value[constraint_mask] - self._atom_offsets[i]
                        elif isinstance(value, torch.Tensor):
                            # Non-index tensor (like bond_lengths when not handled above)
                            sample_topology[key] = value
                
                if sample_topology:
                    data['topology'] = sample_topology
            
            # Handle global features
            global_fields = ['global_features', 'time', 'global_labels', 'global_loss_weights']
            for field_name in global_fields:
                attr = getattr(self, field_name, None)
                if attr is not None:
                    data[field_name] = attr[i:i+1]  # Keep batch dimension
            
            protein_data_list.append(ProteinData(**data))
        
        return protein_data_list
    
    def __len__(self) -> int:
        """Return batch size."""
        return self.batch_size
    
    def __repr__(self) -> str:
        """Return string representation of batch."""
        return f"BatchProteinData(batch_size={self.batch_size}, total_atoms={len(self.batch)})"