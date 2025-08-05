# metalsitenn/dataloading.py
'''
* Author: Evan Komp
* Created: 6/19/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from collections import Counter
import gc
import warnings
from contextlib import contextmanager
import signal
import gzip
import lzma
from functools import lru_cache

from metalsitenn.utils import ParamsObj
from metalsitenn.placer_modules.cifutils import CIFParser
from metalsitenn.constants import I2E, RESNAME_3LETTER

logger = logging.getLogger(__name__)


@contextmanager
def timeout_handler(seconds=300):  # 5 min timeout per file
    """Context manager to timeout stuck files."""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"File processing timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_entity_type(res_name: str, atom_key: tuple, site_chain: 'Chain') -> str:
    """
    Determine entity type from residue name and check if residue contains metals.
    
    Args:
        res_name: 3-letter residue name
        atom_key: Original atom key tuple (chain_id, res_num, res_name, atom_name)
        site_chain: Site-specific Chain object with renumbered residues
        
    Returns:
        Entity type: 'water', 'amino_acid', 'nucleotide', 'metal_ligand', or 'organic_ligand'
    """
    if res_name == 'HOH':
        return 'water'
    elif res_name in RESNAME_3LETTER:
        return 'amino_acid'
    elif res_name in ['A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DG', 'DC']:
        return 'nucleotide'
    else:
        # Check if this residue contains any metal atoms
        for site_atom_key, site_atom in site_chain.atoms.items():
            # site_atom_key is (chain_id, res_num, res_name, atom_name)
            if site_atom_key[2] == res_name and site_atom.metal:
                return 'metal_ligand'
        
        return 'organic_ligand'


def _process_cif_worker(cif_path: Path, parser_params: Dict, 
                       save_pdb: bool, edquality_cache_path: Optional[Path],
                       edquality_csv_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Worker function with per-worker EDQuality mapper initialization.
    
    Args:
        cif_path: Path to CIF file to process
        parser_params: Dictionary of parsing parameters
        save_pdb: Whether to save PDB files
        edquality_cache_path: Path to EDQuality mapper cache
        edquality_csv_path: Path to EDQuality CSV file
        
    Returns:
        List of site metadata dictionaries
    """
    # Initialize mapper once per worker if needed
    if edquality_cache_path and not hasattr(_process_cif_worker, '_edquality_mapper'):
        from metalsitenn.edquality import EDQualityMapping
        _process_cif_worker._edquality_mapper = EDQualityMapping(
            csv_path=edquality_csv_path,
            cache_path=edquality_cache_path
        )
        logger.debug(f"Worker {os.getpid()} initialized EDQuality mapper")
    
    # Add mapper to parser_params for this worker
    if hasattr(_process_cif_worker, '_edquality_mapper'):
        parser_params = parser_params.copy()  # Don't modify original
        parser_params['edquality_mapper'] = _process_cif_worker._edquality_mapper
    
    # Delegate to the implementation
    return _process_single_cif_impl(cif_path, parser_params, save_pdb)


def _process_single_cif_impl(cif_path: Path, parser_params: Dict, 
                            save_pdb: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single CIF file and return metadata (implementation).
    
    Args:
        cif_path: Path to CIF file
        parser_params: Dictionary of parsing parameters
        save_pdb: Whether to save PDB files
        
    Returns:
        List of site metadata dictionaries and saves aggregated sites
    """
    pdb_code = cif_path.stem
    
    try:
        with timeout_handler(300):  # 5 minute timeout
            # Create parser instance for this worker
            parser = CIFParser(skip_res=parser_params.get('skip_entities', []))
            
            # Parse CIF file
            chains, assemblies, covalent, meta = parser.parse(str(cif_path))
            
            # Extract metal sites with EDQuality mapper
            metal_sites = parser.get_metal_sites(
                (chains, assemblies, covalent, meta),
                cutoff_distance=parser_params.get('metal_site_radius', 5.0),
                coordination_distance=parser_params.get('coordination_distance', 3.0),
                merge_threshold=parser_params.get('metal_aggregation_distance', 10.0),
                max_atoms_per_site=parser_params.get('max_atoms_per_site', None),
                min_amino_acids=parser_params.get('min_residues_per_site', None),
                min_coordinating_amino_acids=parser_params.get('min_coordinating_amino_acids', None),
                skip_sites_with_entities=parser_params.get('skip_sites_with_entities', None),
                max_water_bfactor=parser_params.get('max_water_bfactor', None),
                backbone_treatment=parser_params.get('backbone_treatment', 'bound'),
                edqualitymapper=parser_params.get('edquality_mapper', None),
            )
            
            if not metal_sites:
                return []
            
            # Process each site and collect for aggregated saving
            site_metadata_list = []
            sites_data = {}
            
            for i, site_data in enumerate(metal_sites):
                metadata = _extract_site_metadata(site_data, pdb_code, i, meta)
                
                # Skip if no amino acids involved
                if metadata['n_amino_acids'] == 0:
                    continue
                
                site_name = f"{pdb_code}_{i}"  # Keep old naming convention
                sites_data[site_name] = site_data['site_chain']
                
                # Save PDB file if requested
                if save_pdb:
                    pdb_dir = parser_params.get('cache_folder', Path('.')) / "pdbs"
                    pdb_dir.mkdir(parents=True, exist_ok=True)
                    pdb_path = pdb_dir / f"{site_name}.pdb"
                    
                    if site_data['site_chain'] and site_data['site_chain'].atoms:
                        parser.save(site_data['site_chain'], str(pdb_path))
                
                site_metadata_list.append(metadata)
            
            # Save aggregated sites for this PDB (using cache_folder from params)
            if sites_data and 'cache_folder' in parser_params:
                _save_sites_by_pdb(pdb_code, sites_data, parser_params['cache_folder'],
                                 parser_params.get('compression', 'lzma'))
            
            return site_metadata_list
            
    except TimeoutError:
        logger.warning(f"Timeout processing {cif_path}")
        return []
    except Exception as e:
        logger.error(f"Error processing {cif_path}: {e}")
        raise


def _extract_site_metadata(site_data: Dict, pdb_code: str, site_idx: int, meta: Any) -> Dict[str, Any]:
    """
    Extract metadata from a metal binding site for dataset filtering and analysis.
    Combines old API fields with new metadata additions.
    
    Args:
        site_data: Site data dictionary from CIFParser
        pdb_code: PDB code
        site_idx: Site index within PDB
        meta: Metadata object from CIF parsing
        
    Returns:
        Metadata dictionary with both old and new fields
    """
    site_chain = site_data['site_chain']
    site_name = f"{pdb_code}_{site_idx}"
    
    # Count entities by type (old approach)
    entity_counts = Counter()
    residue_names = set()
    non_residue_non_metal_entities = set()
    
    # Process nearby residues (old method)
    for res_key in site_data['nearby_residues']:
        res_name = res_key[2]  # residue name is at index 2
        residue_names.add(res_name)
        entity_type = get_entity_type(res_name, res_key, site_chain)
        entity_counts[entity_type] += 1
        
        # Track non-residue, non-metal entities
        if entity_type not in ['amino_acid', 'nucleotide', 'metal_ligand', 'water']:
            non_residue_non_metal_entities.add(res_name)
    
    # Count coordinating amino acids (old method)
    coordinating_amino_acids = 0
    for res_key in site_data['coordinating_residues']:
        res_name = res_key[2]  # residue name is at index 2
        if res_name in RESNAME_3LETTER:
            coordinating_amino_acids += 1
    coordinating_resids = [res_key[1] for res_key in site_data['coordinating_residues']]
    
    # Get metal information (old method)
    metal_elements = []
    for metal_atom in site_data['metal_atoms']:
        element_symbol = I2E.get(metal_atom['element'], 'UNK')
        metal_elements.append(element_symbol)
    
    unique_metals = sorted(set(metal_elements))
    
    # Count bonds in site chain (old method)
    n_bonds = len(site_data['site_chain'].bonds) if site_data['site_chain'] else 0
    
    # NEW: Track resolution and RCZD (from new version)
    resolution = meta.get('resolution', None) if meta else None
    
    max_rczd = site_data.get('max_rczd', None)
    
    # Combine old fields with new metadata
    return {
        # Old API fields (keep these for backward compatibility)
        'pdb_code': pdb_code,
        'site_name': site_name,
        'site_idx': site_idx,
        'n_entities': len(site_data['nearby_residues']),
        'n_atoms': site_data['n_atoms'],
        'n_bonds': n_bonds,
        'metal': ','.join(unique_metals),
        'n_metals': site_data['n_metals'],
        'n_waters': entity_counts.get('water', 0),
        'n_organic_ligands': entity_counts.get('organic_ligand', 0),
        'n_metal_ligands': entity_counts.get('metal_ligand', 0),
        'n_amino_acids': entity_counts.get('amino_acid', 0),
        'n_coordinating_amino_acids': coordinating_amino_acids,
        'n_nucleotides': entity_counts.get('nucleotide', 0),
        'non_residue_non_metal_names': ','.join(sorted(non_residue_non_metal_entities)) if non_residue_non_metal_entities else '',
        'n_non_residue_non_metal': len(non_residue_non_metal_entities),
        'coordination_distance': site_data.get('coordination_distance', 3.0),
        'n_unresolved_removed': site_data.get('n_unresolved_removed', None),
        'coordinating_residues': ','.join(coordinating_resids),
        
        # NEW: Additional metadata fields
        'resolution': resolution,
        'max_rczd': max_rczd,
    }


def _save_sites_by_pdb(pdb_code: str, sites_data: Dict[str, Any], cache_folder: Path, compression: str):
    """Save all sites for a PDB code to aggregated file."""
    aggregated_folder = Path(cache_folder) / "aggregated_sites"
    aggregated_folder.mkdir(exist_ok=True)
    
    extension = {
        'gzip': '.pkl.gz',
        'lzma': '.pkl.xz', 
        'none': '.pkl'
    }[compression]
    
    output_file = aggregated_folder / f"{pdb_code}_sites{extension}"
    
    if compression == 'gzip':
        with gzip.open(output_file, 'wb', compresslevel=6) as f:
            pickle.dump(sites_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif compression == 'lzma':
        with lzma.open(output_file, 'wb', preset=6) as f:
            pickle.dump(sites_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(sites_data, f, protocol=pickle.HIGHEST_PROTOCOL)


class MetalSiteDataset:
    """
    Optimized PyTorch Dataset for metal binding sites from protein structures.
    
    Features aggregated, compressed cache files with LRU caching for better DVC performance
    while maintaining all original filtering and processing capabilities.
    """
    
    def __init__(self,
                # Data source parameters
                cif_folder: Optional[str] = None,
                cache_folder: str = "cache",
                overwrite: bool = False,
                save_pdb: bool = False,
                
                # Cache optimization parameters
                compression: str = 'lzma',  # 'gzip', 'lzma', 'none'
                max_loaded_pdbs: int = 50,  # LRU cache size
                
                # CIF parsing parameters  
                metal_site_radius: float = 5.0,
                coordination_distance: float = 2.5,
                metal_aggregation_distance: float = 10.0,
                max_atoms_per_site: Optional[int] = None,
                min_residues_per_site: Optional[int] = None,
                min_coordinating_amino_acids: Optional[int] = None,
                skip_sites_with_entities: Optional[Union[List[str], str]] = None,
                max_water_bfactor: Optional[float] = None,
                backbone_treatment: str = 'bound',
                skip_entities: Optional[List[str]] = None,
                
                # EDQuality mapper integration (NEW)
                edquality_mapper = None,
                
                # Multiprocessing
                n_cores: int = 1,
                
                # Filtering parameters (keeping old names + new ones)
                valid_pdb_codes: Optional[List[str]] = None,
                valid_metals: Optional[List[str]] = None,
                min_metals: Optional[int] = None,
                max_metals: Optional[int] = None,
                min_organic_ligands: Optional[int] = None,
                max_organic_ligands: Optional[int] = None,
                min_waters: Optional[int] = None,
                max_waters: Optional[int] = None,
                min_nucleotides: Optional[int] = None,
                max_nucleotides: Optional[int] = None,
                min_amino_acids: Optional[int] = None,
                max_amino_acids: Optional[int] = None,
                min_co_amino_acids: Optional[int] = None,
                max_co_amino_acids: Optional[int] = None,
                min_sites_per_pdb: Optional[int] = None,
                max_sites_per_pdb: Optional[int] = None,
                max_unresolved_removed: Optional[int] = None,
                
                # NEW: Additional filtering parameters
                max_resolution: Optional[float] = None,
                max_max_rczd: Optional[float] = None,

                # Debug parameter
                debug_max_files: Optional[int] = None
                ):
        """
        Initialize the metal site dataset with optimized caching.
        
        Args:
            cif_folder: Path to folder containing CIF files. If provided, will parse from scratch.
            cache_folder: Path to cache directory for pickles and metadata.
            overwrite: If True, overwrite existing cache when parsing from scratch.
            save_pdb: Whether to save PDB files of sites during parsing.
            compression: Compression method ('gzip', 'lzma', 'none').
            max_loaded_pdbs: Maximum PDB files to keep in memory (LRU cache).
            
            # CIF parsing parameters (ignored if loading from cache)
            metal_site_radius: Distance cutoff for including residues around metals.
            coordination_distance: Distance cutoff for coordination analysis.
            metal_aggregation_distance: Distance threshold for merging nearby metals.
            max_atoms_per_site: Maximum heavy atoms per binding site.
            min_residues_per_site: Minimum amino acid residues required in site.
            min_coordinating_amino_acids: Minimum amino acid residues within coordination distance.
            skip_sites_with_entities: Entity names to avoid in sites.
            max_water_bfactor: Maximum B-factor for water oxygen atoms.
            backbone_treatment: Treatment of backbone atoms ('bound', 'free', 'ca_only').
            skip_entities: List of entity names to remove from library.
            edquality_mapper: EDQualityMapping instance for quality assessment.
            
            n_cores: Number of CPU cores for parallel processing (-1 for all).
            
            # Filtering parameters
            valid_pdb_codes: List of valid PDB codes to include.
            min_metals/max_metals: Filter by number of metals per site.
            min_organic_ligands/max_organic_ligands: Filter by organic ligands per site.
            min_waters/max_waters: Filter by number of waters per site.
            min_nucleotides/max_nucleotides: Filter by nucleotides per site.
            min_amino_acids/max_amino_acids: Filter by amino acids per site.
            min_co_amino_acids/max_co_amino_acids: Filter by coordinating amino acids per site.
            min_sites_per_pdb/max_sites_per_pdb: Filter by total sites per PDB.
            max_unresolved_removed: Filter by number of unresolved atoms removed.
            max_resolution: Maximum crystal structure resolution (Angstroms).
            max_max_rczd: Maximum electron density quality score (max RCZD in site).
        """
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.aggregated_folder = self.cache_folder / "aggregated_sites"
        self.aggregated_folder.mkdir(exist_ok=True)
        
        # Cache optimization settings
        self.compression = compression
        self.max_loaded_pdbs = max_loaded_pdbs
        
        # LRU cache will be initialized after parsing to avoid pickling issues
        self._load_pdb_sites = None
        
        # Store EDQuality mapper (NEW)
        self.edquality_mapper = edquality_mapper
        
        # Store filtering parameters
        self.valid_pdb_codes = set(valid_pdb_codes) if valid_pdb_codes else None
        self.valid_metals = set(valid_metals) if valid_metals else None
        self.min_metals = min_metals
        self.max_metals = max_metals
        self.min_organic_ligands = min_organic_ligands
        self.max_organic_ligands = max_organic_ligands
        self.min_waters = min_waters
        self.max_waters = max_waters
        self.min_nucleotides = min_nucleotides
        self.max_nucleotides = max_nucleotides
        self.min_amino_acids = min_amino_acids
        self.max_amino_acids = max_amino_acids
        self.min_coordinating_amino_acids = min_coordinating_amino_acids
        self.max_co = max_co_amino_acids
        self.min_co = min_co_amino_acids
        self.min_sites_per_pdb = min_sites_per_pdb
        self.max_sites_per_pdb = max_sites_per_pdb
        self.max_unresolved_removed = max_unresolved_removed
        self._debug_max_files = debug_max_files
        
        # NEW: Additional filtering parameters
        self.max_resolution = max_resolution
        self.max_max_rczd = max_max_rczd
        
        if cif_folder is not None:
            # Parse from scratch
            self._parse_and_cache(
                cif_folder=cif_folder,
                overwrite=overwrite,
                save_pdb=save_pdb,
                parser_params={
                    'metal_site_radius': metal_site_radius,
                    'coordination_distance': coordination_distance,
                    'metal_aggregation_distance': metal_aggregation_distance,
                    'max_atoms_per_site': max_atoms_per_site,
                    'min_residues_per_site': min_residues_per_site,
                    'min_coordinating_amino_acids': min_coordinating_amino_acids,
                    'skip_sites_with_entities': skip_sites_with_entities,
                    'max_water_bfactor': max_water_bfactor,
                    'backbone_treatment': backbone_treatment,
                    'skip_entities': skip_entities or [],
                    'edquality_mapper': edquality_mapper,
                    'cache_folder': self.cache_folder,
                    'compression': self.compression,
                },
                n_cores=n_cores
            )
        else:
            # Load from cache
            if not (self.cache_folder / "metadata.csv").exists():
                raise FileNotFoundError(f"No metadata.csv found in {self.cache_folder}. "
                                    f"Please provide cif_folder to parse from scratch.")
            logger.info("Loading from existing cache (parsing parameters ignored)")
        
        # Load and filter metadata
        self._load_and_filter_metadata()
        
        # Build fast lookup index (old API style)
        self._build_site_index()
        
        # Initialize LRU cache after parsing (to avoid pickling issues)
        self._init_lru_cache()
        
    def _parse_and_cache(self, cif_folder: str, overwrite: bool, save_pdb: bool,
                        parser_params: Dict, n_cores: int):
        """Parse CIF files and cache results in optimized format."""
        # Check if cache exists and handle overwrite
        metadata_path = self.cache_folder / "metadata.csv"
        if metadata_path.exists() and not overwrite:
            raise FileExistsError(f"Cache already exists at {self.cache_folder}. "
                                f"Set overwrite=True to overwrite.")
        
        # Get CIF files
        cif_path = Path(cif_folder)
        cif_files = list(cif_path.glob("*.cif")) + list(cif_path.glob("*.cif.gz"))
        if self._debug_max_files is not None:
            cif_files = cif_files[:self._debug_max_files]
            logger.info(f"Debug mode: limiting to {self._debug_max_files} CIF files")
        
        if not cif_files:
            raise FileNotFoundError(f"No CIF files found in {cif_folder}")
        
        logger.info(f"Found {len(cif_files)} CIF files to process")
        
        # Handle EDQuality mapper serialization (NEW)
        edquality_cache_path = None
        edquality_csv_path = None
        if 'edquality_mapper' in parser_params and parser_params['edquality_mapper'] is not None:
            # Save mapper to cache if it doesn't exist
            mapper = parser_params['edquality_mapper']
            edquality_cache_path = self.cache_folder / "edquality_mapper.pkl"
            edquality_csv_path = str(mapper.csv_path)  # Store original CSV path
            
            if not edquality_cache_path.exists():
                mapper.cache_path = edquality_cache_path
                mapper.save_to_cache()
            
            # Remove from parser_params to avoid serialization
            del parser_params['edquality_mapper']
        
        # Determine number of cores
        if n_cores == -1:
            n_cores = multiprocessing.cpu_count()
        
        logger.info(f"Processing with {n_cores} cores")
        
        # Process files in parallel
        if n_cores == 1:
            # Sequential processing - load mapper directly
            if edquality_cache_path:
                from metalsitenn.edquality import EDQualityMapping
                parser_params['edquality_mapper'] = EDQualityMapping(
                    csv_path=edquality_csv_path,
                    cache_path=edquality_cache_path
                )
            
            all_metadata = []
            for cif_file in tqdm(cif_files, desc="Processing CIF files"):
                metadata_list = _process_single_cif_impl(
                    cif_file, parser_params, save_pdb
                )
                all_metadata.extend(metadata_list)
        else:
            # Parallel processing with worker initialization
            results = Parallel(n_jobs=n_cores, backend='loky', verbose=1)(
                delayed(_process_cif_worker)(
                    cif_file, parser_params, save_pdb, edquality_cache_path, edquality_csv_path
                ) for cif_file in cif_files
            )
            
            # Flatten results
            all_metadata = []
            for metadata_list in results:
                all_metadata.extend(metadata_list)
        
        # Save metadata
        if all_metadata:
            df = pd.DataFrame(all_metadata)
            df.to_csv(metadata_path, index=False)
            logger.info(f"Saved metadata for {len(df)} sites to {metadata_path}")
        else:
            logger.warning("No metal sites found in any CIF files")
            # Create empty metadata file
            pd.DataFrame().to_csv(metadata_path, index=False)
    
    def _load_and_filter_metadata(self):
        """Load metadata and apply filters (OLD API style)."""
        metadata_path = self.cache_folder / "metadata.csv"
        self.metadata_df = pd.read_csv(metadata_path)
        
        if len(self.metadata_df) == 0:
            logger.warning("No sites found in metadata")
            self.valid_indices = []
            self.filtered_metadata = pd.DataFrame()
            return
        
        # Apply filters (keeping old logic)
        mask = pd.Series([True] * len(self.metadata_df))
        
        if self.valid_pdb_codes is not None:
            mask &= self.metadata_df['pdb_code'].isin(self.valid_pdb_codes)
        
        if self.valid_metals is not None:
            mask &= self.metadata_df['metal'].apply(
                lambda x: any(metal in self.valid_metals for metal in x.split(',') if metal)
            )
        
        if self.min_metals is not None:
            mask &= self.metadata_df['n_metal_ligands'] >= self.min_metals
        if self.max_metals is not None:
            mask &= self.metadata_df['n_metal_ligands'] <= self.max_metals
            
        if self.min_organic_ligands is not None:
            mask &= self.metadata_df['n_organic_ligands'] >= self.min_organic_ligands
        if self.max_organic_ligands is not None:
            mask &= self.metadata_df['n_organic_ligands'] <= self.max_organic_ligands
            
        if self.min_waters is not None:
            mask &= self.metadata_df['n_waters'] >= self.min_waters
        if self.max_waters is not None:
            mask &= self.metadata_df['n_waters'] <= self.max_waters
            
        if self.min_nucleotides is not None:
            mask &= self.metadata_df['n_nucleotides'] >= self.min_nucleotides
        if self.max_nucleotides is not None:
            mask &= self.metadata_df['n_nucleotides'] <= self.max_nucleotides
            
        if self.min_amino_acids is not None:
            mask &= self.metadata_df['n_amino_acids'] >= self.min_amino_acids
        if self.max_amino_acids is not None:
            mask &= self.metadata_df['n_amino_acids'] <= self.max_amino_acids
            
        if self.min_coordinating_amino_acids is not None:
            mask &= self.metadata_df['n_coordinating_amino_acids'] >= self.min_coordinating_amino_acids
            
        if self.min_co is not None:
            mask &= self.metadata_df['n_coordinating_amino_acids'] >= self.min_co
        if self.max_co is not None:
            mask &= self.metadata_df['n_coordinating_amino_acids'] <= self.max_co

        if self.max_unresolved_removed is not None:
            mask &= self.metadata_df['n_unresolved_removed'].fillna(0) <= self.max_unresolved_removed
        
        # NEW: Additional filters
        if self.max_resolution is not None:
            mask &= (~self.metadata_df['resolution'].isna() & 
                    (self.metadata_df['resolution'] <= self.max_resolution))
        
        if self.max_max_rczd is not None:
            mask &= (~self.metadata_df['max_rczd'].isna() & 
                    (self.metadata_df['max_rczd'] <= self.max_max_rczd))
        
        # Filter by sites per PDB
        if self.min_sites_per_pdb is not None or self.max_sites_per_pdb is not None:
            sites_per_pdb = self.metadata_df['pdb_code'].value_counts()
            valid_pdbs = sites_per_pdb.index
            
            if self.min_sites_per_pdb is not None:
                valid_pdbs = valid_pdbs[sites_per_pdb >= self.min_sites_per_pdb]
            if self.max_sites_per_pdb is not None:
                valid_pdbs = valid_pdbs[sites_per_pdb <= self.max_sites_per_pdb]
                
            mask &= self.metadata_df['pdb_code'].isin(valid_pdbs)
        
        self.valid_indices = mask[mask].index.tolist()
        self.filtered_metadata = self.metadata_df.iloc[self.valid_indices].copy()
        
        logger.info(f"Filtered to {len(self.valid_indices)} sites from {len(self.metadata_df)} total")
    
    def _init_lru_cache(self):
        """Initialize LRU cache after construction to avoid pickling issues."""
        self._load_pdb_sites = lru_cache(maxsize=self.max_loaded_pdbs)(self._load_pdb_sites_uncached)
    
    def _build_site_index(self):
        """Build fast lookup index from site_name to PDB code (OLD API style)."""
        self._site_to_pdb_index = {
            f"{row['pdb_code']}_{row['site_idx']}": row['pdb_code']
            for _, row in self.metadata_df.iterrows()
        }
    
    # Optimized cache methods
    def _get_aggregated_filename(self, pdb_code: str) -> Path:
        """Get filename for aggregated sites file."""
        extension = {
            'gzip': '.pkl.gz',
            'lzma': '.pkl.xz', 
            'none': '.pkl'
        }[self.compression]
        
        return self.aggregated_folder / f"{pdb_code}_sites{extension}"
    
    def _load_pdb_sites_uncached(self, pdb_code: str) -> Dict[str, Any]:
        """Load all sites for a PDB (uncached version for LRU wrapper)."""
        aggregated_file = self._get_aggregated_filename(pdb_code)
        
        if not aggregated_file.exists():
            raise FileNotFoundError(f"No aggregated file for PDB {pdb_code}")
        
        if self.compression == 'gzip':
            with gzip.open(aggregated_file, 'rb') as f:
                return pickle.load(f)
        elif self.compression == 'lzma':
            with lzma.open(aggregated_file, 'rb') as f:
                return pickle.load(f)
        else:
            with open(aggregated_file, 'rb') as f:
                return pickle.load(f)
    
    def _load_site(self, site_name: str) -> Any:
        """Load individual site from aggregated cache with LRU caching."""
        # Ensure LRU cache is initialized
        if self._load_pdb_sites is None:
            self._init_lru_cache()
            
        if site_name not in self._site_to_pdb_index:
            raise KeyError(f"Site {site_name} not found in index")
            
        pdb_code = self._site_to_pdb_index[site_name]
        sites_data = self._load_pdb_sites(pdb_code)
        
        if site_name not in sites_data:
            raise KeyError(f"Site {site_name} not found in PDB {pdb_code} data")
            
        return sites_data[site_name]
    
    # Standard Dataset interface methods (OLD API)
    def __len__(self) -> int:
        """Return number of valid sites after filtering."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        """
        Get a metal binding site (OLD API format).
        
        Args:
            idx: Index of the site to retrieve.
            
        Returns:
            Tuple of (site_name, Chain object) where site_name is "{pdb_code}_{site_idx}".
        """
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get metadata for this site
        metadata_idx = self.valid_indices[idx]
        site_info = self.metadata_df.iloc[metadata_idx]
        
        # Load site from optimized cache
        site_name = f"{site_info['pdb_code']}_{site_info['site_idx']}"
        site_chain = self._load_site(site_name)
        
        return site_name, site_chain
    
    def get_site_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific site by index (OLD API)."""
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        metadata_idx = self.valid_indices[idx]
        return self.metadata_df.iloc[metadata_idx].to_dict()
    
    def get_filtered_metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame for valid sites after filtering (OLD API)."""
        return self.filtered_metadata.copy()
    
    def get_all_metadata(self) -> pd.DataFrame:
        """Get complete metadata DataFrame before filtering (OLD API)."""
        return self.metadata_df.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics (OLD API name)."""
        # Ensure LRU cache is initialized
        if self._load_pdb_sites is None:
            self._init_lru_cache()
            
        cache_info = self._load_pdb_sites.cache_info()
        return {
            'lru_hits': cache_info.hits,
            'lru_misses': cache_info.misses,
            'lru_hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if cache_info.hits + cache_info.misses > 0 else 0,
            'loaded_pdbs': cache_info.currsize,
            'max_loaded_pdbs': cache_info.maxsize,
            'total_sites_indexed': len(self._site_to_pdb_index),
            'filtered_sites': len(self.valid_indices),
            'compression': self.compression
        }
    
    # Additional utility methods for backward compatibility
    def clear_cache(self):
        """Clear the LRU cache to free memory."""
        if self._load_pdb_sites is not None:
            self._load_pdb_sites.cache_clear()
            gc.collect()
    
    def save_filtered_metadata(self, output_path: str):
        """Save filtered metadata to CSV."""
        self.filtered_metadata.to_csv(output_path, index=False)
        logger.info(f"Saved filtered metadata ({len(self.filtered_metadata)} sites) to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if len(self.filtered_metadata) == 0:
            return {"n_sites": 0, "n_pdbs": 0}
        
        # Count metals
        metal_counts = Counter()
        for metal_str in self.filtered_metadata['metal']:
            if pd.notna(metal_str) and metal_str:
                for metal in metal_str.split(','):
                    if metal:
                        metal_counts[metal] += 1
        
        # Basic statistics
        stats = {
            'n_sites': len(self.filtered_metadata),
            'n_pdbs': self.filtered_metadata['pdb_code'].nunique(),
            'avg_atoms_per_site': self.filtered_metadata['n_atoms'].mean(),
            'avg_entities_per_site': self.filtered_metadata['n_entities'].mean(),
            'avg_waters_per_site': self.filtered_metadata['n_waters'].mean(),
            'avg_metals_per_site': self.filtered_metadata['n_metals'].mean(),
            'avg_organic_ligands_per_site': self.filtered_metadata['n_organic_ligands'].mean(),
            'avg_metal_ligands_per_site': self.filtered_metadata['n_metal_ligands'].mean(),
            'avg_amino_acids_per_site': self.filtered_metadata['n_amino_acids'].mean(),
            'avg_coordinating_amino_acids_per_site': self.filtered_metadata['n_coordinating_amino_acids'].mean(),
        }
        
        # Add metal counts
        for metal, count in metal_counts.items():
            stats[f'n_{metal}'] = count
        
        # Resolution statistics if available (NEW)
        if 'resolution' in self.filtered_metadata.columns:
            resolution_data = self.filtered_metadata['resolution'].dropna()
            if len(resolution_data) > 0:
                stats.update({
                    'avg_resolution': resolution_data.mean(),
                    'min_resolution': resolution_data.min(),
                    'max_resolution': resolution_data.max(),
                })
        
        # RCZD statistics if available (NEW)
        if 'max_rczd' in self.filtered_metadata.columns:
            rczd_data = self.filtered_metadata['max_rczd'].dropna()
            if len(rczd_data) > 0:
                stats.update({
                    'avg_max_rczd': rczd_data.mean(),
                    'min_max_rczd': rczd_data.min(),
                    'max_max_rczd': rczd_data.max(),
                })
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        stats = self.get_statistics()
        return f"MetalSiteDataset(n_sites={stats['n_sites']}, n_pdbs={stats['n_pdbs']}, cache_folder='{self.cache_folder}')"