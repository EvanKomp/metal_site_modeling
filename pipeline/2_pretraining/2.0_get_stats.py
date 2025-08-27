# pipeline/2_pretraining/2.0_get_node_element_stats.py
'''
* Author: Evan Komp
* Created: 8/21/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import dvc.api
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metalsitenn.utils import ParamsObj, make_jsonable
from metalsitenn.dataloading import MetalSiteDataset
from metalsitenn.featurizer import MetalSiteCollator

logger = logging.getLogger(__name__)

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_element_counts(dataloader: DataLoader, collator: MetalSiteCollator, max_sites=None) -> Dict[str, Dict]:
    """
    Extract element counts from dataset using collator.
    
    Args:
        dataloader: DataLoader for the dataset
        collator: MetalSiteCollator instance
        
    Returns:
        Dictionary with 'element_strings' and 'element_tokens' counts
    """
    element_tokenizer = collator.featurizer.tokenizers['element']
    element_string_counts = Counter()
    element_token_counts = Counter()
    
    logger.info("Processing dataset to extract element counts...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            logger.info(f"Processed {batch_idx} batches...")
            
        # Count element tokens
        element_tokens = batch.element.flatten().tolist()
        for token_id in element_tokens:
            element_token_counts[token_id] += 1
            
        # Decode tokens to element strings and count them
        element_strings = element_tokenizer.decode_sequence(element_tokens)
        for element_str in element_strings:
            element_string_counts[element_str] += 1

        if max_sites is not None and batch_idx >= max_sites:
            logger.info(f"Reached max_sites limit: {max_sites}")
            break
    
    return {
        'element_strings': dict(element_string_counts),
        'element_tokens': element_token_counts
    }

def compute_dataset_stats(dataset: MetalSiteDataset, filtered_metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics.
    
    Args:
        dataset: MetalSiteDataset instance
        filtered_metadata: Filtered metadata DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    original_metadata = dataset.get_all_metadata()
    
    stats = {
        'original_dataset_size': len(original_metadata),
        'final_dataset_size': len(filtered_metadata),
        'filtering_retention_rate': len(filtered_metadata) / len(original_metadata) if len(original_metadata) > 0 else 0.0
    }
    
    if len(filtered_metadata) > 0:
        # Basic counts and averages
        stats.update({
            'n_unique_pdbs': filtered_metadata['pdb_code'].nunique(),
            'avg_atoms_per_site': float(filtered_metadata['n_atoms'].mean()),
            'avg_entities_per_site': float(filtered_metadata['n_entities'].mean()),
            'avg_waters_per_site': float(filtered_metadata['n_waters'].mean()),
            'avg_metals_per_site': float(filtered_metadata['n_metals'].mean()),
            'avg_organic_ligands_per_site': float(filtered_metadata['n_organic_ligands'].mean()),
            'avg_metal_ligands_per_site': float(filtered_metadata['n_metal_ligands'].mean()),
            'avg_amino_acids_per_site': float(filtered_metadata['n_amino_acids'].mean()),
            'avg_coordinating_amino_acids_per_site': float(filtered_metadata['n_coordinating_amino_acids'].mean()),
        })
        
        # Resolution statistics if available
        if 'resolution' in filtered_metadata.columns:
            resolution_data = filtered_metadata['resolution'].dropna()
            if len(resolution_data) > 0:
                stats.update({
                    'avg_resolution': float(resolution_data.mean()),
                    'min_resolution': float(resolution_data.min()),
                    'max_resolution': float(resolution_data.max()),
                    'n_sites_with_resolution': len(resolution_data)
                })
        
        # RCZD statistics if available
        if 'max_rczd' in filtered_metadata.columns:
            rczd_data = filtered_metadata['max_rczd'].dropna()
            if len(rczd_data) > 0:
                stats.update({
                    'avg_max_rczd': float(rczd_data.mean()),
                    'min_max_rczd': float(rczd_data.min()),
                    'max_max_rczd': float(rczd_data.max()),
                    'n_sites_with_rczd': len(rczd_data)
                })
        
        # Metal distribution
        metal_counts = Counter()
        for metal_str in filtered_metadata['metal']:
            if pd.notna(metal_str) and metal_str:
                for metal in metal_str.split(','):
                    if metal.strip():
                        metal_counts[metal.strip()] += 1
        
        stats['metal_distribution'] = dict(metal_counts)
        
        # Sites per PDB distribution
        sites_per_pdb = filtered_metadata['pdb_code'].value_counts()
        stats.update({
            'avg_sites_per_pdb': float(sites_per_pdb.mean()),
            'min_sites_per_pdb': int(sites_per_pdb.min()),
            'max_sites_per_pdb': int(sites_per_pdb.max())
        })
    
    return stats

def main():
    """Main function to compute dataset statistics."""
    # Load parameters
    PARAMS_ = dvc.api.params_show()
    PARAMS = ParamsObj(PARAMS_)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/2.0_get_stats.log')
    
    logger.info("Starting dataset statistics computation")
    logger.info(f"Filtering params: {PARAMS_['2_pretraining']['data']['filtering']}")
    logger.info(f"Tokenization params: {PARAMS_['2_pretraining']['data']['tokenization']}")
    
    # Create output directories
    output_dir = Path('data/2/2.0')
    metrics_dir = Path('data/metrics/2/2.0')
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset with filtering parameters
    cache_folder = 'data/1/1.1_parse_sites_metadata'
    filtering_params = PARAMS_['2_pretraining']['data']['filtering']
    
    try:
        dataset = MetalSiteDataset(
            cache_folder=cache_folder,
            overwrite=False,
            save_pdb=False,
            
            # Apply filtering parameters
            **filtering_params
        )
        
        logger.info(f"Dataset initialized with {len(dataset)} filtered sites")
        
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        raise
    
    # Get metadata for statistics
    original_metadata = dataset.get_all_metadata()
    filtered_metadata = dataset.get_filtered_metadata()
    
    logger.info(f"Original dataset size: {len(original_metadata)}")
    logger.info(f"Filtered dataset size: {len(filtered_metadata)}")
    
    # Compute basic dataset statistics
    dataset_stats = compute_dataset_stats(dataset, filtered_metadata)
    logger.info("Computed basic dataset statistics")
    
    # Initialize collator for featurization
    tokenization_params = PARAMS_['2_pretraining']['data']['tokenization']
    dataloader_n_processes = PARAMS_['2_pretraining']['data']['dataloader_n_processes']
    
    collator = MetalSiteCollator(
        atom_features=tokenization_params['atom_features'],
        bond_features=tokenization_params['bond_features'],
        # No masking, collapsing, or metal classification for stats
        metal_classification=False,
        residue_collapse_do=False,
        node_mlm_do=False,
        active_aggregators=tokenization_params['active_aggregators']
    )
    
    logger.info(f"Initialized collator with features: {tokenization_params['atom_features']} + {tokenization_params['bond_features']}")
    logger.info(f"Active aggregators: {tokenization_params['active_aggregators']}")
    
    # Create dataloader for processing
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        num_workers=dataloader_n_processes,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Created dataloader with {dataloader_n_processes} workers")
    
    # Extract element counts
    try:
        element_counts = extract_element_counts(dataloader, collator, max_sites=PARAMS_['2_pretraining']['data']['debug_max_sites'])
        logger.info("Extracted element counts from dataset")
        
        # Save element counts
        element_counts_path = output_dir / 'element_counts.json'
        with open(element_counts_path, 'w') as f:
            json.dump(element_counts, f, indent=2)
        
        logger.info(f"Saved element counts to {element_counts_path}")
        
        # Add vocabulary sizes to dataset stats
        dataset_stats['vocab_sizes'] = collator.featurizer.get_feature_vocab_sizes()
        dataset_stats['unique_element_strings'] = len(element_counts['element_strings'])
        
    except Exception as e:
        logger.error(f"Failed to extract element counts: {e}")
        raise
    
    # Save dataset statistics
    stats_path = metrics_dir / 'data_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    logger.info(f"Saved dataset statistics to {stats_path}")
    
    # Log summary
    logger.info("=" * 50)
    logger.info("DATASET STATISTICS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Original dataset size: {dataset_stats['original_dataset_size']:,}")
    logger.info(f"Final dataset size: {dataset_stats['final_dataset_size']:,}")
    logger.info(f"Retention rate: {dataset_stats['filtering_retention_rate']:.2%}")
    logger.info(f"Unique PDBs: {dataset_stats.get('n_unique_pdbs', 'N/A'):,}")
    logger.info(f"Avg atoms per site: {dataset_stats.get('avg_atoms_per_site', 'N/A'):.1f}")
    logger.info(f"Avg coordinating residues: {dataset_stats.get('avg_coordinating_amino_acids_per_site', 'N/A'):.1f}")
    
    if 'avg_resolution' in dataset_stats:
        logger.info(f"Avg resolution: {dataset_stats['avg_resolution']:.2f} Å")
    
    if 'metal_distribution' in dataset_stats:
        top_metals = sorted(dataset_stats['metal_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top metals: {', '.join([f'{m}({c})' for m, c in top_metals])}")
    
    logger.info("Dataset statistics computation completed successfully!")

if __name__ == "__main__":
    main()