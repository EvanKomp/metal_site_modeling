# pipeline/1.2_create_dataset.py
'''
* Author: Evan Komp
* Created: 11/26/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Creates HuggingFace dataset from PDB files with tokenized atomic data.

Params:
    data.model_hydrogens (bool): Include hydrogen atoms
    data.metal_known (bool): Use unique tokens for metals vs generic METAL token

Inputs:
    data/mf_sites/*.pdb: PDB format files containing metal binding sites

Outputs:
    data/dataset/metal_site_dataset/: HuggingFace dataset directory
'''

import os
import logging
from typing import Generator, Dict, Any

import datasets
from datasets import Dataset
import dvc.api
import joblib

from metalsitenn.data import PDBReader
from metalsitenn.atom_vocabulary import AtomTokenizer
from metalsitenn.utils import ParamsObj

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='logs/1.2_create_dataset.log')

def load_params() -> ParamsObj:
    """Load DVC params."""
    return ParamsObj(dvc.api.params_show())

def data_generator(
    reader: PDBReader,
    data_dir: str,
    debug_sample: int = None
) -> Generator[Dict[str, Any], None, None]:
    """Yields examples from PDB directory.
    
    Args:
        reader: Configured PDB reader
        data_dir: Directory containing PDB files
        
    Yields:
        Dict containing atomic data for each structure
    """
    for i, example in enumerate(reader.read_dir(data_dir)):
        # Convert numpy arrays to lists for HF dataset
        example['pos'] = example['pos'].tolist()
        yield example
        if debug_sample and i >= debug_sample:
            break

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load params
    params = load_params()

    # load tokenizer if present
    changed = True
    if os.path.exists('data/dataset/tokenizer.pkl'):
        changed=False
        tokenizer = joblib.load('data/dataset/tokenizer.pkl')

        # skip the recreation by loading and concatenating 
        # first check that the only thing about params changed is test frac
        data_params = dvc.api.params_show(stages="1.2_create_dataset")['data']
        for key, value in data_params.items():
            if hasattr(tokenizer, key) and getattr(tokenizer, key) != value:
                changed = True
                break
    if changed:

        # Initialize components
        reader = PDBReader(deprotonate=not params.data.model_hydrogens)
        tokenizer = AtomTokenizer(
            keep_hydrogen=params.data.model_hydrogens,
            metal_known=params.data.metal_known,
            aggregate_uncommon=params.data.aggregate_uncommon,
            allow_unknown=True
        )
        # save the toknizer
        joblib.dump(tokenizer, 'data/dataset/tokenizer.pkl')

        logger.info("Creating dataset from PDB files...")
        dataset = Dataset.from_generator(
            lambda: data_generator(reader, "data/mf_sites", debug_sample=None)
        )

        # Add tokens
        logger.info("Tokenizing atomic data...")
        def tokenize(example):
            tokens = tokenizer.tokenize(
                example['atoms'],
                example['atom_types']
            )
            example.update(tokens)
            return example
            
        dataset = dataset.map(tokenize, desc="Tokenizing atomic data")

    else:
        logger.info("Loading dataset from disk...")
        dataset = datasets.load_from_disk("data/dataset/metal_site_dataset")
        # this is a data dict, concat so we can re-split
        dataset = datasets.concatenate_datasets([dataset[d] for d in dataset.keys()])

    # split into train, test, and validation sets
    dataset = dataset.train_test_split(test_size=params.data.test_frac)
    
    # Save dataset
    # first save to tmp directory, then move
    out_dir = "data/dataset/metal_site_dataset"
    logger.info(f"Saving dataset to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir+"_tmp")
    os.rename(out_dir+"_tmp", out_dir)
    
    logger.info(f"Created dataset with {len(dataset['train'])} training examples, {len(dataset['test'])} test examples")

if __name__ == "__main__":
    main()