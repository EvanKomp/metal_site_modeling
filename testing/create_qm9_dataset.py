# testing/create_qm9_dataset.py
'''
* Author: Evan Komp
* Created: 4/30/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import logging
import torch
import numpy as np
import datasets
import joblib
from torch_geometric.datasets import QM9
from tqdm import tqdm

from metalsitenn.atom_vocabulary import UnknownTokenError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='logs/create_qm9_dataset.log')

# QM9 element mapping: atomic numbers to element symbols
QM9_ELEMENTS = {
    1: 'H',
    6: 'C', 
    7: 'N',
    8: 'O',
    9: 'F'
}

def process_qm9_data(data_obj, keep_hydrogen=False):
    """
    Converts a PyTorch Geometric QM9 data point to the format expected by MetalSiteNN
    
    Args:
        data_obj: torch_geometric.data.Data object from QM9
        keep_hydrogen: Whether to include hydrogen atoms
        
    Returns:
        Dict containing atoms, atom_types, positions and id
    """
    # Get element for each atom from atomic number (Z)
    atom_nums = data_obj.z.cpu().numpy()
    atoms = [QM9_ELEMENTS[int(z)] for z in atom_nums]
    
    # Get positions
    pos = data_obj.pos.cpu().numpy().tolist()
    
    # Filter out hydrogens if needed
    if not keep_hydrogen:
        mask = atom_nums != 1
        atoms = [a for i, a in enumerate(atoms) if mask[i]]
        pos = [p for i, p in enumerate(pos) if mask[i]]
    
    # All QM9 atoms are treated as HETATM for consistency
    atom_types = ['HETATM'] * len(atoms)
    
    return {
        'atoms': atoms,
        'atom_types': atom_types,
        'pos': pos,
        'id': f'qm9_{int(data_obj.idx)}',
        'y': data_obj.y.cpu().numpy().tolist()  # Store all 19 target properties
    }

def data_generator(tokenizer, max_samples=10000, keep_hydrogen=False):
    """
    Generate QM9 molecules in the format needed for MetalSiteNN
    
    Args:
        tokenizer: Initialized AtomTokenizer
        max_samples: Maximum number of samples to include
        keep_hydrogen: Whether to include hydrogen atoms
        
    Yields:
        Formatted molecule examples
    """
    # Load QM9 dataset
    root = os.path.join('data', 'qm9_raw')
    dataset = QM9(root=root)
    logger.info(f"Loaded QM9 dataset with {len(dataset)} molecules")
    
    # Process molecules
    total_processed = 0
    total_valid = 0
    total_tokenization_errors = 0
    
    # Use a random order
    indices = np.random.permutation(len(dataset))
    
    for idx in tqdm(indices):
        data_obj = dataset[idx]
        
        # Process data
        result = process_qm9_data(data_obj, keep_hydrogen=keep_hydrogen)
        
        try:
            # Tokenize
            tokens = tokenizer.tokenize([result['atoms']], [result['atom_types']])
            
            # Convert tokens to lists if necessary
            if isinstance(tokens['atoms'][0], torch.Tensor):
                atom_tokens = tokens['atoms'][0].tolist()
            else:
                atom_tokens = tokens['atoms'][0]
                
            if isinstance(tokens['atom_types'][0], torch.Tensor):
                type_tokens = tokens['atom_types'][0].tolist()
            else:
                type_tokens = tokens['atom_types'][0]
            
            # Check consistency
            if not (len(atom_tokens) == len(type_tokens) == len(result['pos'])):
                logger.warning(f"Inconsistent lengths in molecule {result['id']}")
                continue
            
            # Update result with tokenized values
            result['atoms'] = atom_tokens
            result['atom_types'] = type_tokens
            
            total_valid += 1
            yield result
            
            # Check if we have enough samples
            if total_valid >= max_samples:
                break
                
        except UnknownTokenError as e:
            logger.debug(f"Unknown token in {result['id']}: {str(e)}")
            total_tokenization_errors += 1
            continue
        except Exception as e:
            logger.error(f"Error processing {result['id']}: {str(e)}")
            total_tokenization_errors += 1
            continue
        
        total_processed += 1
        if total_processed % 1000 == 0:
            logger.info(f"Processed {total_processed} molecules, found {total_valid} valid, {total_tokenization_errors} errors")

def main():
    """Create a dataset from QM9 molecules."""
    logger.info("Starting QM9 dataset creation")
    
    # Load tokenizer
    tokenizer = joblib.load("data/dataset/tokenizer.pkl")
    logger.info(f"Loaded tokenizer with atom vocab size: {tokenizer.atom_vocab.vocab_size}")
    
    # Create dataset
    logger.info("Creating dataset from QM9 molecules...")
    dataset = datasets.Dataset.from_generator(
        lambda: data_generator(tokenizer, max_samples=10000, keep_hydrogen=False)
    )
    
    # Final consistency check
    def check_data_integrity(example):
        lengths = [len(example[k]) for k in ['atoms', 'atom_types', 'pos']]
        return len(set(lengths)) == 1
    
    valid_count = sum(1 for ex in dataset if check_data_integrity(ex))
    logger.info(f"Final dataset contains {valid_count}/{len(dataset)} valid examples")
    
    if valid_count < len(dataset):
        filtered_dataset = dataset.filter(check_data_integrity)
        dataset = filtered_dataset
        logger.info(f"Filtered dataset to {len(dataset)} examples")
    
    # Split into train/test
    splits = dataset.train_test_split(test_size=0.1)
    
    # Save to disk
    splits.save_to_disk("data/qm9_dataset")
    logger.info(f"Dataset saved to disk with {len(splits['train'])} training and {len(splits['test'])} test examples")

if __name__ == "__main__":
    main()