# pipeline/1.0_construct_edquality_mapper.py
'''
* Author: Evan Komp
* Created: 7/30/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import os
import logging
from pathlib import Path
import dvc.api

from metalsitenn.edquality import EDQualityMapping
from metalsitenn.utils import ParamsObj


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
    """Main function to construct EDQualityMapper and save to cache."""
    # Load parameters
    PARAMS = dvc.api.params_show()
    PARAMS = ParamsObj(PARAMS)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging('logs/1.0_construct_edquality_mapper.log')
    
    logger.info("Starting EDQualityMapper construction")
    logger.info(f"Data source: {PARAMS.data.source}")
    
    # Define input and output paths
    csv_path = "data/edstats_pdb_quality.csv"
    cif_folder = PARAMS.data.source
    cache_path = "data/edquality_mapper_cache"
    
    # Ensure cache directory exists
    os.makedirs(Path(cache_path).parent, exist_ok=True)
    
    logger.info(f"CSV path: {csv_path}")
    logger.info(f"CIF folder: {cif_folder}")
    logger.info(f"Cache path: {cache_path}")
    
    try:
        # Construct EDQualityMapping with n_jobs=-1 (use all cores)
        ed_mapper = EDQualityMapping(
            csv_path=csv_path,
            cif_folder=cif_folder,
            cache_path=cache_path,
            coordinate_tolerance=0.5,
            n_jobs=-1
        )
        
        # Get stats for logging
        stats = ed_mapper.get_statistics()
        logger.info(f"EDQualityMapper construction complete:")
        logger.info(f"  Total entries: {stats['total_entries']}")
        logger.info(f"  Unique PDBs: {stats['unique_pdbs']}")
        logger.info(f"  RSZD mean: {stats.get('rszd_mean', 'N/A'):.3f}")
        logger.info(f"  RSZD std: {stats.get('rszd_std', 'N/A'):.3f}")
        logger.info(f"  Metal counts: {stats.get('metal_counts', {})}")
        
        logger.info(f"Cache saved to: {cache_path}")
        
    except Exception as e:
        logger.error(f"Failed to construct EDQualityMapper: {e}")
        raise


if __name__ == "__main__":
    main()