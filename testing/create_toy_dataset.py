# testing/create_toy_dataset.py
'''
* Author: Evan Komp
* Created: 2/12/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import torch
import numpy as np
from e3nn import o3
import datasets

import joblib

def get_bond_length(atom1: str, atom2: str) -> float:
    """Returns ideal bond length in angstroms."""
    bond_lengths = {
        ('<METAL>', 'O'): 1.6,
        ('<METAL>', 'N'): 2.1, 
        ('<METAL>', 'S'): 2.0,
        ('<METAL>', 'P'): 2.3,
        ('C', 'O'): 1.2,
        ('C', 'N'): 1.3,
        ('C', 'C'): 1.5,
        ('C', 'S'): 1.8,
        ('C', 'P'): 1.8,
    }
    # Sort to ensure consistent lookup
    key = tuple(sorted([atom1, atom2]))
    return bond_lengths.get(key, 2.0)  # Default to 2.0

def get_geometry_vectors(geom_type: str) -> torch.Tensor:
    """Returns unit vectors for different geometries."""
    if geom_type == "linear2":
        return torch.tensor([[0,0,0], [0,0,-1]])
    elif geom_type == "linear3":
        return torch.tensor([[0,0,0],[0,0,1], [0,0,-1]])
    elif geom_type == "tetrahedral":
        return torch.tensor([
            [0,0,0],
            [0,0,1],
            [0.942809,0,-0.333333],
            [-0.471405,0.816497,-0.333333],
            [-0.471405,-0.816497,-0.333333]
        ])
    elif geom_type == "octahedral":
        return torch.tensor([
            [0,0,0],
            [0,0,1],
            [0,0,-1],
            [1,0,0],
            [-1,0,0],
            [0,1,0],
            [0,-1,0]
        ])
    else:
        raise ValueError(f"Unknown geometry: {geom_type}")

def generate_metal_site():
    """Generates a random metal site with ideal geometry."""
    # Choose geometry
    geom_type = np.random.choice(["linear2", "linear3", "tetrahedral", "octahedral"])
    vectors = get_geometry_vectors(geom_type)
    
    # Choose atoms
    ligands = np.random.choice(['O', 'N', 'S', 'P'], size=len(vectors)-1)
    atoms = ['<METAL>'] + list(ligands)
    
    # some stupid rule about atom type based on identity
    atom_types = []
    for atom in atoms:
        if atom in ['O']:
            atom_types.append('HETATM')
        else:
            atom_types.append('ATOM')
    
    # Generate positions
    positions = []
    for i, (atom, vec) in enumerate(zip(atoms, vectors)):
        if i == 0:
            pos = vec
        else:
            bond_length = get_bond_length('<METAL>', atom)
            pos = vec * bond_length
        positions.append(pos)
    
    # Random rotation
    R = o3.rand_matrix()
    positions = torch.stack(positions) @ R.T

    # random translation
    translation = torch.rand(3) * 10.0
    positions += translation
    
    return {
        'pos': positions,
        'atoms': atoms,
        'atom_types': atom_types,
    }

def main():
    tokenizer = joblib.load("data/dataset/tokenizer.pkl")

    def metal_site_generator():
        for i in range(100000):
            out_dict = generate_metal_site()
            out_dict.update(tokenizer.tokenize(out_dict['atoms'], out_dict['atom_types']))
            yield out_dict

    ds = datasets.Dataset.from_generator(metal_site_generator)
    # split
    dd = ds.train_test_split(test_size=0.1)
    dd.save_to_disk("data/toy_dataset")

if __name__ == "__main__":
    main()
