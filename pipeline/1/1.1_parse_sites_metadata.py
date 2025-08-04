# pipeline/1/1.1_parse_sites_metadata.py
'''
* Author: Evan Komp
* Created: 6/16/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from collections import Counter
import dvc.api
import multiprocessing

from metalsitenn.utils import ParamsObj, make_jsonable
from metalsitenn.dataloading import MetalSiteDataset
from metalsitenn.edquality import EDQualityMapping

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

def main():
    """Main function to parse metal sites metadata."""
    # Load parameters
    PARAMS = dvc.api.params_show()
    PARAMS = ParamsObj(PARAMS)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/1.1_parse_site_metadata.log')
    
    logger.info("Starting metal sites metadata parsing")
    logger.info(f"Data source: {PARAMS.data.source}")
    logger.info(f"Metal site radius: {PARAMS.data.metal_site_radius}")
    logger.info(f"Metal aggregation distance: {PARAMS.data.metal_aggregation_distance}")
    
    # Determine number of cores to use
    n_cores = getattr(PARAMS.data, 'n_cores', -1)
    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()
    
    logger.info(f"Using {n_cores} CPU cores for parallel processing")
    
    # Create output directory
    output_dir = 'data/1/1.1_parse_sites_metadata'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset (this will do the parsing and caching)
    try:
        # load the edmapper
        mapper = EDQualityMapping(
            csv_path='./data/edstats_pdb_quality.csv',
            cache_path='./data/edquality_mapper_cache',
            cif_folder=PARAMS.data.source,
            coordinate_tolerance=0.1)


        dataset = MetalSiteDataset(
            cif_folder=PARAMS.data.source,
            cache_folder=output_dir,
            overwrite=True,
            save_pdb=False,  # No need to save PDBs for metadata generation
            
            # CIF parsing parameters
            metal_site_radius=PARAMS.data.metal_site_radius,
            metal_aggregation_distance=PARAMS.data.metal_aggregation_distance,
            max_atoms_per_site=PARAMS.data.max_atoms_per_site,
            min_residues_per_site=PARAMS.data.min_residues_per_site,
            coordination_distance=PARAMS.data.coordination_distance,
            min_coordinating_amino_acids=PARAMS.data.min_coordinating_residues_per_site,
            skip_sites_with_entities=PARAMS.data.skip_sites_with_entities,
            max_water_bfactor=PARAMS.data.max_water_bfactor,
            backbone_treatment=PARAMS.data.backbone_treatment,
            skip_entities=PARAMS.data.remove_entities,

            # for getting rczd
            edquality_mapper=mapper,
            
            n_cores=n_cores,
            
            # Apply debug file limit if specified 
            debug_max_files=PARAMS.data.debug_max_files
        )
        
        logger.info(f"Dataset initialized with {len(dataset)} sites")
        
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        raise
    
    # Get metadata
    metadata_df = dataset.get_all_metadata()
    
    if len(metadata_df) == 0:
        logger.warning("No metal sites found in any CIF files")
        return
    
    # The metadata CSV is already saved by the dataset, but we'll also save it 
    # in the expected location for backward compatibility
    output_csv = os.path.join(output_dir, 'metal_sites_metadata.csv')
    metadata_df.to_csv(output_csv, index=False)
    logger.info(f"Saved metadata for {len(metadata_df)} sites to {output_csv}")
    
    # Generate aggregate metrics
    metal_counts = Counter()
    for metal_str in metadata_df['metal']:
        if pd.notna(metal_str) and metal_str:  # Handle NaN and empty strings
            for metal in metal_str.split(','):
                if metal:
                    metal_counts[metal] += 1
    
    processed_pdbs = metadata_df['pdb_code'].nunique()
    
    aggregate_metrics = {
        'n_sites': len(metadata_df),
        'n_pdbs': processed_pdbs,
        'avg_atoms_per_site': metadata_df['n_atoms'].mean(),
        'avg_entities_per_site': metadata_df['n_entities'].mean(),
        'avg_waters_per_site': metadata_df['n_waters'].mean(),
        'avg_metals_per_site': metadata_df['n_metals'].mean(),
        'avg_organic_ligands_per_site': metadata_df['n_organic_ligands'].mean(),
        'avg_metal_ligands_per_site': metadata_df['n_metal_ligands'].mean(),
        'avg_amino_acids_per_site': metadata_df['n_amino_acids'].mean(),
    }
    
    # Add metal counts as flat keys
    for metal, count in metal_counts.items():
        aggregate_metrics[f'n_{metal}'] = count
    
    # Save aggregate metrics
    aggregate_metrics = make_jsonable(aggregate_metrics)
    output_json = os.path.join(output_dir, 'aggregate_metrics.json')
    with open(output_json, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    logger.info(f"Saved aggregate metrics to {output_json}")
    logger.info(f"Processing complete: {aggregate_metrics['n_sites']} sites from {aggregate_metrics['n_pdbs']} PDBs")

if __name__ == "__main__":
    main()