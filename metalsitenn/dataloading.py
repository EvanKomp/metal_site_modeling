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


# Add to process_single_cif_for_dataset
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
        List of site metadata dictionaries
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
            
            # Process each site and collect metadata
            site_metadata_list = []
            
            for i, site_data in enumerate(metal_sites):
                metadata = _extract_site_metadata(site_data, pdb_code, i)
                
                # Skip if no amino acids involved
                if metadata['n_amino_acids'] == 0:
                    continue
                
                site_id = f"{pdb_code}_{i}"
                site_metadata_list.append(metadata)
                
                # Save site data if requested
                if save_pdb:
                    _save_site_pdb(site_data, site_id, parser_params.get('cache_folder'))
            
            return site_metadata_list
            
    except TimeoutError:
        logger.warning(f"Timeout processing {cif_path}")
        return []
    except Exception as e:
        raise


def _extract_site_metadata(site_data: Dict, pdb_code: str, site_index: int) -> Dict[str, Any]:
    """
    Extract metadata from a metal site.
    
    Args:
        site_data: Site data dictionary from CIFParser
        pdb_code: PDB code
        site_index: Site index within PDB
        
    Returns:
        Metadata dictionary
    """
    site_chain = site_data['site_chain']
    
    # Count entities by type
    entity_counts = Counter()
    metal_list = []
    coordinating_amino_acids = 0
    max_rczd = None
    
    for atom_key, atom in site_chain.atoms.items():
        res_name = atom_key[2]
        entity_type = get_entity_type(res_name, atom_key, site_chain)
        entity_counts[entity_type] += 1
        
        # Track metals
        if atom.metal:
            metal_list.append(I2E.get(atom.element, str(atom.element)))
            
        # Track coordinating amino acids (within coordination distance)
        if entity_type == 'amino_acid' and hasattr(atom, 'coordinating') and atom.coordinating:
            coordinating_amino_acids += 1
            
        # Track maximum RCZD if available
        if hasattr(atom, 'rczd') and atom.rczd is not None:
            if max_rczd is None or atom.rczd > max_rczd:
                max_rczd = atom.rczd
    
    # Get resolution if available
    resolution = getattr(site_data.get('meta', {}), 'resolution', None)
    
    return {
        'pdb_code': pdb_code,
        'site_index': site_index,
        'site_id': f"{pdb_code}_{site_index}",
        'n_atoms': len(site_chain.atoms),
        'n_entities': len(set(atom_key[2] for atom_key in site_chain.atoms.keys())),
        'n_waters': entity_counts.get('water', 0),
        'n_amino_acids': entity_counts.get('amino_acid', 0),
        'n_nucleotides': entity_counts.get('nucleotide', 0),
        'n_organic_ligands': entity_counts.get('organic_ligand', 0),
        'n_metal_ligands': entity_counts.get('metal_ligand', 0),
        'n_metals': len(metal_list),
        'metal': ','.join(sorted(set(metal_list))) if metal_list else '',
        'n_coordinating_amino_acids': coordinating_amino_acids,
        'max_rczd': max_rczd,
        'resolution': resolution,
    }


def _save_site_pdb(site_data: Dict, site_id: str, cache_folder: Optional[Path]):
    """
    Save site as PDB file.
    
    Args:
        site_data: Site data dictionary
        site_id: Unique site identifier
        cache_folder: Cache folder path
    """
    if cache_folder is None:
        return
        
    pdb_folder = Path(cache_folder) / "pdbs"
    pdb_folder.mkdir(exist_ok=True)
    
    pdb_path = pdb_folder / f"{site_id}.pdb"
    
    try:
        site_chain = site_data['site_chain']
        with open(pdb_path, 'w') as f:
            for atom_key, atom in site_chain.atoms.items():
                # Write ATOM/HETATM record
                record_type = "HETATM" if atom.metal else "ATOM  "
                f.write(f"{record_type}{atom.serial:5d} {atom.name:4s} {atom_key[2]:3s} "
                       f"{atom_key[0]:1s}{atom_key[1]:4s}    "
                       f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                       f"{getattr(atom, 'occupancy', 1.0):6.2f}"
                       f"{getattr(atom, 'b_factor', 0.0):6.2f}          "
                       f"{atom.element:2s}\n")
    except Exception as e:
        logger.warning(f"Failed to save PDB for {site_id}: {e}")


class MetalSiteDataset:
    """
    Optimized PyTorch Dataset for metal binding sites from protein structures.
    
    Features aggregated, compressed cache files with LRU caching for better DVC performance
    while maintaining all original filtering and processing capabilities with efficient
    EDQuality mapper handling for multiprocessing.
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
                
                # EDQuality mapper integration
                edquality_mapper = None,
                
                # Multiprocessing
                n_cores: int = 1,
                
                # Filtering parameters
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
        
        # Store EDQuality mapper
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
        self.max_resolution = max_resolution
        self.max_max_rczd = max_max_rczd
        self._debug_max_files = debug_max_files
        
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
        
        # Build fast lookup index
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
        
        # Handle EDQuality mapper serialization
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
        """Load metadata CSV and apply filtering criteria."""
        metadata_path = self.cache_folder / "metadata.csv"
        self.metadata_df = pd.read_csv(metadata_path)
        
        if len(self.metadata_df) == 0:
            logger.warning("No sites found in metadata")
            return
        
        initial_count = len(self.metadata_df)
        
        # Apply filters
        if self.valid_pdb_codes:
            self.metadata_df = self.metadata_df[
                self.metadata_df['pdb_code'].isin(self.valid_pdb_codes)
            ]
        
        if self.valid_metals:
            metal_mask = self.metadata_df['metal'].apply(
                lambda x: bool(x) and any(metal in self.valid_metals for metal in x.split(','))
            )
            self.metadata_df = self.metadata_df[metal_mask]
        
        # Numeric filters
        numeric_filters = [
            ('n_metals', self.min_metals, self.max_metals),
            ('n_organic_ligands', self.min_organic_ligands, self.max_organic_ligands),
            ('n_waters', self.min_waters, self.max_waters),
            ('n_nucleotides', self.min_nucleotides, self.max_nucleotides),
            ('n_amino_acids', self.min_amino_acids, self.max_amino_acids),
            ('n_coordinating_amino_acids', self.min_co, self.max_co),
            ('resolution', None, self.max_resolution),
            ('max_rczd', None, self.max_max_rczd),
        ]
        
        for column, min_val, max_val in numeric_filters:
            if min_val is not None:
                self.metadata_df = self.metadata_df[self.metadata_df[column] >= min_val]
            if max_val is not None:
                self.metadata_df = self.metadata_df[
                    self.metadata_df[column].isna() | (self.metadata_df[column] <= max_val)
                ]
        
        # Sites per PDB filter
        if self.min_sites_per_pdb or self.max_sites_per_pdb:
            sites_per_pdb = self.metadata_df.groupby('pdb_code').size()
            valid_pdbs = sites_per_pdb.index
            
            if self.min_sites_per_pdb:
                valid_pdbs = sites_per_pdb[sites_per_pdb >= self.min_sites_per_pdb].index
            if self.max_sites_per_pdb:
                valid_pdbs = sites_per_pdb[sites_per_pdb <= self.max_sites_per_pdb].index
                
            self.metadata_df = self.metadata_df[
                self.metadata_df['pdb_code'].isin(valid_pdbs)
            ]
        
        logger.info(f"Filtered to {len(self.metadata_df)} sites from {initial_count} total")
    
    def _build_site_index(self):
        """Build fast lookup index for sites."""
        self.site_id_to_index = {
            row['site_id']: idx for idx, row in self.metadata_df.iterrows()
        }
        self.pdb_to_sites = {}
        for _, row in self.metadata_df.iterrows():
            pdb_code = row['pdb_code']
            if pdb_code not in self.pdb_to_sites:
                self.pdb_to_sites[pdb_code] = []
            self.pdb_to_sites[pdb_code].append(row['site_id'])
    
    def _init_lru_cache(self):
        """Initialize LRU cache after parsing to avoid pickling issues."""
        @lru_cache(maxsize=self.max_loaded_pdbs)
        def load_pdb_sites(pdb_code: str) -> Dict[str, Any]:
            """Load sites for a PDB from cache with LRU eviction."""
            return self._load_pdb_sites_impl(pdb_code)
        
        self._load_pdb_sites = load_pdb_sites
    
    def _load_pdb_sites_impl(self, pdb_code: str) -> Dict[str, Any]:
        """Implementation of PDB site loading."""
        cache_file = self.aggregated_folder / f"{pdb_code}.pkl"
        
        if self.compression == 'gzip':
            cache_file = cache_file.with_suffix('.pkl.gz')
        elif self.compression == 'lzma':
            cache_file = cache_file.with_suffix('.pkl.xz')
        
        if not cache_file.exists():
            logger.warning(f"Cache file {cache_file} not found")
            return {}
        
        try:
            if self.compression == 'gzip':
                with gzip.open(cache_file, 'rb') as f:
                    return pickle.load(f)
            elif self.compression == 'lzma':
                with lzma.open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache file {cache_file}: {e}")
            return {}
    
    def get_all_metadata(self) -> pd.DataFrame:
        """
        Get metadata for all sites in dataset.
        
        Returns:
            DataFrame with metadata for all sites
        """
        return self.metadata_df.copy()
    
    def get_site_metadata(self, site_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific site.
        
        Args:
            site_id: Site identifier (format: "pdb_code_site_index")
            
        Returns:
            Metadata dictionary for the site
        """
        if site_id not in self.site_id_to_index:
            raise KeyError(f"Site {site_id} not found in dataset")
        
        idx = self.site_id_to_index[site_id]
        return self.metadata_df.iloc[idx].to_dict()
    
    def get_site_structure(self, site_id: str) -> Any:
        """
        Get the parsed structure for a specific site.
        
        Args:
            site_id: Site identifier (format: "pdb_code_site_index")
            
        Returns:
            Site chain object with atomic coordinates
        """
        pdb_code = site_id.split('_')[0]
        
        if self._load_pdb_sites is None:
            raise RuntimeError("Dataset not properly initialized")
        
        pdb_sites = self._load_pdb_sites(pdb_code)
        
        if site_id not in pdb_sites:
            raise KeyError(f"Site {site_id} not found in cached structures")
        
        return pdb_sites[site_id]
    
    def __len__(self) -> int:
        """Get number of sites in dataset."""
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get site data by index.
        
        Args:
            idx: Site index
            
        Returns:
            Dictionary containing site metadata and structure
        """
        if idx >= len(self.metadata_df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        site_row = self.metadata_df.iloc[idx]
        site_id = site_row['site_id']
        
        # Get structure data
        try:
            structure = self.get_site_structure(site_id)
        except KeyError:
            logger.warning(f"Structure not found for site {site_id}")
            structure = None
        
        return {
            'site_id': site_id,
            'metadata': site_row.to_dict(),
            'structure': structure
        }
    
    def get_sites_by_pdb(self, pdb_code: str) -> List[Dict[str, Any]]:
        """
        Get all sites for a specific PDB.
        
        Args:
            pdb_code: PDB code
            
        Returns:
            List of site dictionaries
        """
        if pdb_code not in self.pdb_to_sites:
            return []
        
        sites = []
        for site_id in self.pdb_to_sites[pdb_code]:
            idx = self.site_id_to_index[site_id]
            sites.append(self[idx])
        
        return sites
    
    def get_sites_by_metal(self, metal: str) -> List[Dict[str, Any]]:
        """
        Get all sites containing a specific metal.
        
        Args:
            metal: Metal symbol (e.g., 'ZN', 'MG')
            
        Returns:
            List of site dictionaries
        """
        metal_mask = self.metadata_df['metal'].apply(
            lambda x: bool(x) and metal in x.split(',')
        )
        metal_sites = self.metadata_df[metal_mask]
        
        sites = []
        for _, row in metal_sites.iterrows():
            idx = self.site_id_to_index[row['site_id']]
            sites.append(self[idx])
        
        return sites
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if len(self.metadata_df) == 0:
            return {"n_sites": 0, "n_pdbs": 0}
        
        # Count metals
        metal_counts = Counter()
        for metal_str in self.metadata_df['metal']:
            if pd.notna(metal_str) and metal_str:
                for metal in metal_str.split(','):
                    if metal:
                        metal_counts[metal] += 1
        
        # Basic statistics
        stats = {
            'n_sites': len(self.metadata_df),
            'n_pdbs': self.metadata_df['pdb_code'].nunique(),
            'avg_atoms_per_site': self.metadata_df['n_atoms'].mean(),
            'avg_entities_per_site': self.metadata_df['n_entities'].mean(),
            'avg_waters_per_site': self.metadata_df['n_waters'].mean(),
            'avg_metals_per_site': self.metadata_df['n_metals'].mean(),
            'avg_organic_ligands_per_site': self.metadata_df['n_organic_ligands'].mean(),
            'avg_metal_ligands_per_site': self.metadata_df['n_metal_ligands'].mean(),
            'avg_amino_acids_per_site': self.metadata_df['n_amino_acids'].mean(),
            'avg_coordinating_amino_acids_per_site': self.metadata_df['n_coordinating_amino_acids'].mean(),
        }
        
        # Add metal counts
        for metal, count in metal_counts.items():
            stats[f'n_{metal}'] = count
        
        # Resolution statistics if available
        if 'resolution' in self.metadata_df.columns:
            resolution_data = self.metadata_df['resolution'].dropna()
            if len(resolution_data) > 0:
                stats.update({
                    'avg_resolution': resolution_data.mean(),
                    'min_resolution': resolution_data.min(),
                    'max_resolution': resolution_data.max(),
                })
        
        # RCZD statistics if available
        if 'max_rczd' in self.metadata_df.columns:
            rczd_data = self.metadata_df['max_rczd'].dropna()
            if len(rczd_data) > 0:
                stats.update({
                    'avg_max_rczd': rczd_data.mean(),
                    'min_max_rczd': rczd_data.min(),
                    'max_max_rczd': rczd_data.max(),
                })
        
        return stats
    
    def filter_dataset(self, **filters) -> 'MetalSiteDataset':
        """
        Create a new filtered dataset.
        
        Args:
            **filters: Filtering criteria (same as constructor parameters)
            
        Returns:
            New MetalSiteDataset with applied filters
        """
        # Create new dataset with same cache but different filters
        new_dataset = MetalSiteDataset(
            cache_folder=str(self.cache_folder),
            compression=self.compression,
            max_loaded_pdbs=self.max_loaded_pdbs,
            **filters
        )
        
        return new_dataset
    
    def save_filtered_metadata(self, output_path: str):
        """
        Save filtered metadata to CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        self.metadata_df.to_csv(output_path, index=False)
        logger.info(f"Saved filtered metadata ({len(self.metadata_df)} sites) to {output_path}")
    
    def clear_cache(self):
        """Clear the LRU cache to free memory."""
        if self._load_pdb_sites is not None:
            self._load_pdb_sites.cache_clear()
            gc.collect()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if self._load_pdb_sites is None:
            return {"cache_enabled": False}
        
        cache_info = self._load_pdb_sites.cache_info()
        return {
            "cache_enabled": True,
            "cache_size": cache_info.currsize,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "max_size": cache_info.maxsize,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
        }
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        stats = self.get_statistics()
        return f"MetalSiteDataset(n_sites={stats['n_sites']}, n_pdbs={stats['n_pdbs']}, cache_folder='{self.cache_folder}')"