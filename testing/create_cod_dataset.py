# testing/create_cod_dataset.py
'''
* Author: Evan Komp
* Created: 4/23/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
from pymatgen.io.cif import CifParser
from pathlib import Path
import numpy as np
from metalsitenn.constants import METAL_IONS
import joblib
import datasets
import logging
import threading
import queue
import torch

from metalsitenn.atom_vocabulary import UnknownTokenError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, filemode='w', filename='logs/create_cod_dataset.log')

def parse_cif_in_thread(cif_path, result_queue):
    """Parse CIF file in a thread and put result in queue."""
    try:
        parser = CifParser(cif_path)
        structures = parser.get_structures()
        
        if not structures:
            logger.debug(f"No structures found in {cif_path}")
            result_queue.put(None)
            return
            
        structure = structures[0]
        
        # Check structure constraints
        has_metal = False
        heavy_atom_count = 0
        if len(structure.sites) > 1000:
            logger.debug(f"Structure too large: {len(structure.sites)} sites in {cif_path}")
            result_queue.put(None)
            return
            
        # Extract atomic data - pre-filter hydrogen atoms
        atoms = []
        atom_types = []
        positions = []
        
        for site in structure.sites:
            element = site.species_string.split(':')[0].upper()
            if element == 'H':
                continue
                
            if element in METAL_IONS:
                has_metal = True
                
            heavy_atom_count += 1
            if heavy_atom_count > 100:
                logger.debug(f"Too many heavy atoms: {heavy_atom_count} in {cif_path}")
                result_queue.put(None)
                return
                
            atoms.append(element)
            atom_types.append('HETATM')
            positions.append(site.coords.tolist())
            
        if not has_metal:
            logger.debug(f"No metal atoms found in {cif_path}")
            result_queue.put(None)
            return
            
        # Verify consistency
        if not (len(atoms) == len(atom_types) == len(positions)):
            logger.error(f"Inconsistent lengths in parsed data: atoms={len(atoms)}, atom_types={len(atom_types)}, positions={len(positions)}")
            result_queue.put(None)
            return
            
        result_queue.put({
            'atoms': atoms,
            'atom_types': atom_types,
            'pos': positions,
            'id': str(Path(cif_path).stem),
            'formula': structure.formula
        })
    except Exception as e:
        logger.debug(f"Error parsing {cif_path}: {str(e)}")
        result_queue.put(None)

def parse_cif_with_timeout(cif_path, timeout=5):
    """Parse CIF file with thread-based timeout.
    
    Args:
        cif_path: Path to CIF file
        timeout: Timeout in seconds
        
    Returns:
        Parsed data dict or None
    """
    result_queue = queue.Queue()
    thread = threading.Thread(
        target=parse_cif_in_thread, 
        args=(cif_path, result_queue)
    )
    thread.daemon = True
    thread.start()
    
    try:
        # Wait for result with timeout
        return result_queue.get(timeout=timeout)
    except queue.Empty:
        logger.debug(f"Timeout parsing {cif_path}")
        return None

def data_generator():
    """Generator for dataset creation."""
    tokenizer = joblib.load("data/dataset/tokenizer.pkl")
    max_samples = 10000
    total_processed = 0
    total_valid = 0
    total_tokenization_errors = 0
    
    # Process CIF files
    all_paths = Path("/projects/metalsitenn/cod/cod/cif/").rglob("*.cif")
    np.random.seed(42)
    random_order_paths = np.random.permutation(list(all_paths))
    for cif_path in random_order_paths:
        total_processed += 1
        
        if total_processed % 1000 == 0:
            logger.info(f"Processed {total_processed} files, found {total_valid} valid structures, {total_tokenization_errors} tokenization errors")
            
        if total_valid >= max_samples:
            break
            
        result = parse_cif_with_timeout(cif_path)            
        
        if result:
            if int(result['id']) == 6000749:
                print("Debugging structure 6000749")
            try:
                # Verify input consistency
                if not (len(result['atoms']) == len(result['atom_types']) == len(result['pos'])):
                    logger.error(f"Inconsistent lengths before tokenization: atoms={len(result['atoms'])}, atom_types={len(result['atom_types'])}, positions={len(result['pos'])}")
                    continue
                
                # Debug log original data
                logger.debug(f"Structure {result['id']}: {len(result['atoms'])} atoms")
                logger.debug(f"Sample atoms: {result['atoms'][:5]}")
                logger.debug(f"Sample types: {result['atom_types'][:5]}")
                
                # Tokenize
                try:
                    tokens = tokenizer.tokenize([result['atoms']], [result['atom_types']])
                
                except UnknownTokenError as e:
                    logger.debug(f"Unknown token in {result['id']}: {str(e)}")
                    continue
                
                # Convert tokens to lists if needed
                if isinstance(tokens['atoms'][0], torch.Tensor):
                    atom_tokens = tokens['atoms'][0].tolist()
                else:
                    atom_tokens = tokens['atoms'][0]
                    
                if isinstance(tokens['atom_types'][0], torch.Tensor):
                    type_tokens = tokens['atom_types'][0].tolist()
                else:
                    type_tokens = tokens['atom_types'][0]
                
                # Verify output consistency
                if len(atom_tokens) != len(type_tokens):
                    logger.error(f"Tokenization created inconsistent lengths: atoms={len(atom_tokens)}, atom_types={len(type_tokens)}")
                    total_tokenization_errors += 1
                    continue
                    
                if len(atom_tokens) != len(result['pos']):
                    logger.error(f"Tokenization changed data length: original={len(result['pos'])}, tokenized={len(atom_tokens)}")
                    total_tokenization_errors += 1
                    continue
                
                # Update the result with tokenized values
                result['atoms'] = atom_tokens
                result['atom_types'] = type_tokens
                
                total_valid += 1
                yield result
                
            except Exception as e:
                logger.error(f"Error during tokenization for {result['id']}: {str(e)}")
                total_tokenization_errors += 1
                raise e
                raise ValueError(f"Tokenization error for {result['id']}: inputs: {result['atoms']}, {result['atom_types']}, token outputs: {tokens} - {str(e)}")

def main():
    """Create a dataset from COD CIF files containing metals."""
    logger.info("Starting dataset creation")
    
    # Test the tokenizer first
    tokenizer = joblib.load("data/dataset/tokenizer.pkl")
    logger.info(f"Loaded tokenizer with atom vocab size: {tokenizer.atom_vocab.vocab_size}")
    logger.info(f"Tokenizer allow_unknown: {getattr(tokenizer.atom_vocab, 'allow_unknown', False)}")
    
    # Find a test file
    test_cif = next(Path("/projects/metalsitenn/cod/cod/cif/").rglob("*.cif"))
    test_result = parse_cif_with_timeout(test_cif)
    
    if test_result:
        logger.info("Testing tokenization on a single example...")
        try:
            tokens = tokenizer.tokenize([test_result['atoms']], [test_result['atom_types']])
            logger.info(f"Test tokenization successful: atoms length={len(tokens['atoms'][0])}, original length={len(test_result['atoms'])}")
            if len(tokens['atoms'][0]) != len(test_result['atoms']):
                logger.error("CRITICAL: Tokenizer is changing the length of the data. Stopping execution.")
                return
        except Exception as e:
            logger.error(f"Test tokenization failed: {str(e)}")
            logger.error("Stopping execution due to tokenization failure.")
            return
    else:
        logger.warning("No valid test structure found. Proceeding with caution.")
    
    # Create the full dataset
    logger.info("Creating dataset from generator...")
    dataset = datasets.Dataset.from_generator(data_generator)
    
    # Final consistency check
    def check_data_integrity(example):
        lengths = [len(example[k]) for k in ['atoms', 'atom_types', 'pos']]
        if len(set(lengths)) != 1:
            logger.error(f"Inconsistent lengths in example {example['id']}: {lengths}")
            return False
        return True
    
    valid_count = sum(1 for ex in dataset if check_data_integrity(ex))
    logger.info(f"Final dataset contains {valid_count}/{len(dataset)} valid examples")
    
    if valid_count < len(dataset):
        logger.error("Some examples have inconsistent data lengths after processing.")
        filtered_dataset = dataset.filter(check_data_integrity)
        dataset = filtered_dataset
        logger.info(f"Filtered dataset to {len(dataset)} examples")
    
    # Split into train/test
    splits = dataset.train_test_split(test_size=0.1)
    
    # Save to disk
    splits.save_to_disk("data/toy_dataset")
    logger.info(f"Dataset saved to disk with {len(splits['train'])} training and {len(splits['test'])} test examples")

if __name__ == "__main__":
    main()