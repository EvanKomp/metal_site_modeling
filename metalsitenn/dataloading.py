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
def timeout_handler(seconds=300):  # 5 m timeout per file
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
        # The atom_key contains the original chain/residue info, but we need to check
        # atoms in the site_chain which has been renumbered.
        # We can check by looking for any atom in site_chain with the same residue name
        # that has the metal flag set.
        
        for site_atom_key, site_atom in site_chain.atoms.items():
            # site_atom_key is (chain_id, res_num, res_name, atom_name)
            if site_atom_key[2] == res_name and site_atom.metal:
                return 'metal_ligand'
        
        return 'organic_ligand'


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
        
        # Determine number of cores
        if n_cores == -1:
            n_cores = multiprocessing.cpu_count()
        
        logger.info(f"Processing with {n_cores} cores")
        
        # Process files in parallel
        if n_cores == 1:
            # Sequential processing for debugging
            all_metadata = []
            for cif_file in tqdm(cif_files, desc="Processing CIF files"):
                metadata_list = self._process_single_cif(
                    cif_file, parser_params, save_pdb
                )
                all_metadata.extend(metadata_list)
        else:
            # Parallel processing
            results = Parallel(n_jobs=n_cores, verbose=1)(
                delayed(self._process_single_cif)(
                    cif_file, parser_params, save_pdb
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
    
    def _process_single_cif(self, cif_path: Path, parser_params: Dict, 
                           save_pdb: bool = False) -> List[Dict[str, Any]]:
        """Process a single CIF file and save sites to optimized cache."""
        from metalsitenn.placer_modules.cifutils import CIFParser
        
        pdb_code = cif_path.stem
        
        try:
            # Create parser instance for this worker
            parser = CIFParser(skip_res=parser_params.get('skip_entities', []))
            
            # Parse CIF file
            chains, assemblies, covalent, meta = parser.parse(str(cif_path))
            
            # Extract metal sites
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
            )
            
            if not metal_sites:
                return []
            
            # Process each site and collect for aggregated saving
            site_metadata_list = []
            sites_data = {}
            
            for i, site_data in enumerate(metal_sites):
                metadata = self._extract_site_metadata(site_data, pdb_code, i)
                
                # Skip if no amino acids involved
                if metadata['n_amino_acids'] == 0:
                    continue
                
                site_id = f"{pdb_code}_{i}"
                sites_data[site_id] = site_data['site_chain']
                
                # Save PDB file if requested
                if save_pdb:
                    pdb_dir = self.cache_folder / "pdbs"
                    pdb_dir.mkdir(parents=True, exist_ok=True)
                    pdb_path = pdb_dir / f"{site_id}.pdb"
                    
                    if site_data['site_chain'] and site_data['site_chain'].atoms:
                        parser.save(site_data['site_chain'], str(pdb_path))
                
                site_metadata_list.append(metadata)
            
            # Save aggregated sites for this PDB
            if sites_data:
                self._save_sites_by_pdb(pdb_code, sites_data)
            
            return site_metadata_list
            
        except Exception as e:
            logger.error(f"Error processing {cif_path}: {e}")
            raise
    
    def _extract_site_metadata(self, site_data: Dict, pdb_code: str, site_idx: int) -> Dict[str, Any]:
        """Extract metadata from a metal binding site for dataset filtering and analysis."""
        from metalsitenn.constants import I2E, RESNAME_3LETTER
        
        site_name = f"{pdb_code}_{site_idx}"
        
        # Count entities by type
        entity_counts = Counter()
        residue_names = set()
        non_residue_non_metal_entities = set()
        
        for res_key in site_data['nearby_residues']:
            res_name = res_key[2]  # residue name is at index 2
            residue_names.add(res_name)
            entity_type = get_entity_type(res_name, res_key, site_data['site_chain'])
            entity_counts[entity_type] += 1
            
            # Track non-residue, non-metal entities
            if entity_type not in ['amino_acid', 'nucleotide', 'metal_ligand', 'water']:
                non_residue_non_metal_entities.add(res_name)
        
        # Count coordinating amino acids
        coordinating_amino_acids = 0
        for res_key in site_data['coordinating_residues']:
            res_name = res_key[2]  # residue name is at index 2
            if res_name in RESNAME_3LETTER:
                coordinating_amino_acids += 1
        
        # Get metal information
        metal_elements = []
        for metal_atom in site_data['metal_atoms']:
            element_symbol = I2E.get(metal_atom['element'], 'UNK')
            metal_elements.append(element_symbol)
        
        unique_metals = sorted(set(metal_elements))
        
        # Count bonds in site chain
        n_bonds = len(site_data['site_chain'].bonds) if site_data['site_chain'] else 0
        
        return {
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
            'coordination_distance': site_data.get('coordination_distance', 3.0)
        }
    
    def _load_and_filter_metadata(self):
        """Load metadata and apply filters."""
        metadata_path = self.cache_folder / "metadata.csv"
        self.metadata_df = pd.read_csv(metadata_path)
        
        if len(self.metadata_df) == 0:
            logger.warning("No sites found in metadata")
            self.valid_indices = []
            self.filtered_metadata = pd.DataFrame()
            return
        
        # Apply filters
        mask = pd.Series([True] * len(self.metadata_df))
        
        if self.valid_pdb_codes is not None:
            mask &= self.metadata_df['pdb_code'].isin(self.valid_pdb_codes)
        
        if self.valid_metals is not None:
            mask &= self.metadata_df['metal'].apply(
                lambda x: any(metal in self.valid_metals for metal in x.split(',') if metal)
            )
        
        if self.min_metals is not None:
            mask &= self.metadata_df['n_metals'] >= self.min_metals
        if self.max_metals is not None:
            mask &= self.metadata_df['n_metals'] <= self.max_metals
            
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
        """Build fast lookup index from site_id to PDB code."""
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
    
    def _save_sites_by_pdb(self, pdb_code: str, sites_data: Dict[str, Any]):
        """Save all sites for a PDB code to aggregated file."""
        output_file = self._get_aggregated_filename(pdb_code)
        
        if self.compression == 'gzip':
            with gzip.open(output_file, 'wb', compresslevel=6) as f:
                pickle.dump(sites_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif self.compression == 'lzma':
            with lzma.open(output_file, 'wb', preset=6) as f:
                pickle.dump(sites_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_file, 'wb') as f:
                pickle.dump(sites_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
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
    
    def _load_site(self, site_id: str) -> Any:
        """Load individual site from aggregated cache with LRU caching."""
        # Ensure LRU cache is initialized
        if self._load_pdb_sites is None:
            self._init_lru_cache()
            
        if site_id not in self._site_to_pdb_index:
            raise KeyError(f"Site {site_id} not found in index")
            
        pdb_code = self._site_to_pdb_index[site_id]
        sites_data = self._load_pdb_sites(pdb_code)
        
        if site_id not in sites_data:
            raise KeyError(f"Site {site_id} not found in PDB {pdb_code} data")
            
        return sites_data[site_id]
    
    # Standard Dataset interface methods
    def __len__(self) -> int:
        """Return number of valid sites after filtering."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        """
        Get a metal binding site.
        
        Args:
            idx: Index of the site to retrieve.
            
        Returns:
            Tuple of (site_id, Chain object) where site_id is "{pdb_code}_{site_idx}".
        """
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get metadata for this site
        metadata_idx = self.valid_indices[idx]
        site_info = self.metadata_df.iloc[metadata_idx]
        
        # Load site from optimized cache
        site_id = f"{site_info['pdb_code']}_{site_info['site_idx']}"
        site_chain = self._load_site(site_id)
        
        return site_id, site_chain
    
    def get_site_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific site by index."""
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        metadata_idx = self.valid_indices[idx]
        return self.metadata_df.iloc[metadata_idx].to_dict()
    
    def get_filtered_metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame for valid sites after filtering."""
        return self.filtered_metadata.copy()
    
    def get_all_metadata(self) -> pd.DataFrame:
        """Get complete metadata DataFrame before filtering."""
        return self.metadata_df.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
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