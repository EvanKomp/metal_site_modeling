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

def extract_site_metadata(site_data: Dict, pdb_code: str, site_idx: int) -> Dict[str, Any]:
    """Extract metadata from a metal binding site for dataset filtering and analysis.
    
    Args:
        site_data: Dictionary containing site information from CIF parser
        pdb_code: PDB code for the structure
        site_idx: Index of the site within the structure
        
    Returns:
        Dictionary with site metadata for filtering and analysis
    """
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
        'coordination_distance': site_data.get('coordination_distance', 3.0)  # Store coordination distance used
    }


def process_single_cif_for_dataset(cif_path: Path, cache_folder: Path, parser_params: Dict, 
                                 save_pdb: bool = False) -> List[Dict[str, Any]]:
    """Process a single CIF file and save sites to cache.
    
    Args:
        cif_path: Path to CIF file
        cache_folder: Directory to save processed sites
        parser_params: Parameters for CIF parsing
        save_pdb: Whether to save PDB files of sites
        
    Returns:
        List of site metadata dictionaries
    """
    pdb_code = cif_path.stem
    
    try:
        # Create parser instance for this worker
        with timeout_handler(seconds=60):  # 5 s timeout per file
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
            
            # Process each site
            site_metadata_list = []
            for i, site_data in enumerate(metal_sites):
                metadata = extract_site_metadata(site_data, pdb_code, i)
                
                # Skip if no amino acids involved
                if metadata['n_amino_acids'] == 0:
                    continue
                
                # Save site chain as pickle
                site_filename = f"{pdb_code}_{i}.pkl"
                site_path = cache_folder / "sites" / site_filename
                site_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(site_path, 'wb') as f:
                    pickle.dump(site_data['site_chain'], f)
                
                # Save PDB file if requested
                if save_pdb:
                    pdb_dir = cache_folder / "pdbs"
                    pdb_dir.mkdir(parents=True, exist_ok=True)
                    pdb_path = pdb_dir / f"{pdb_code}_{i}.pdb"
                    
                    if site_data['site_chain'] and site_data['site_chain'].atoms:
                        parser.save(site_data['site_chain'], str(pdb_path))
                
                site_metadata_list.append(metadata)

            del parser
            gc.collect()  # Clean up parser memory
            
            return site_metadata_list

    except TimeoutError as e:
        logger.error(f"TIMEOUT: {pdb_code} after 5 s")
        return []
    except Exception as e:
        warnings.warn(f"Failed to process {cif_path}: {e}")
        return []

class MetalSiteDataset(Dataset):
    """
    PyTorch Dataset for metal binding sites from protein structures.
    
    During initialization, if cif_folder is provided, will parse all CIF files
    and cache the resulting sites as pickles. Otherwise loads from existing cache.
    """
    
    def __init__(self,
                # Data source parameters
                cif_folder: Optional[str] = None,
                cache_folder: str = "cache",
                overwrite: bool = False,
                save_pdb: bool = False,
                
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

                # ignore for production
                debug_max_files: Optional[int] = None
                ):
        """Initialize the metal site dataset.
        
        Args:
            cif_folder: Path to folder containing CIF files. If provided, will parse from scratch.
            cache_folder: Path to cache directory for pickles and metadata.
            overwrite: If True, overwrite existing cache when parsing from scratch.
            save_pdb: Whether to save PDB files of sites during parsing.
            
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
            max_coordinating_amino_acids: Filter by coordinating amino acids per site.
            min_sites_per_pdb/max_sites_per_pdb: Filter by total sites per PDB.
        """
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self._debug_max_files = debug_max_files

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
    
    def _parse_and_cache(self, cif_folder: str, overwrite: bool, save_pdb: bool,
                        parser_params: Dict, n_cores: int):
        """Parse CIF files and cache results."""
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
                metadata_list = process_single_cif_for_dataset(
                    cif_file, self.cache_folder, parser_params, save_pdb
                )
                all_metadata.extend(metadata_list)
        else:
            # Parallel processing
            results = Parallel(n_jobs=n_cores, verbose=1)(
                delayed(process_single_cif_for_dataset)(
                    cif_file, self.cache_folder, parser_params, save_pdb
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
        """Load metadata and apply filters."""
        metadata_path = self.cache_folder / "metadata.csv"
        self.metadata_df = pd.read_csv(metadata_path)
        
        if len(self.metadata_df) == 0:
            logger.warning("No sites found in metadata")
            self.valid_indices = []
            return
        
        # Apply filters
        mask = pd.Series([True] * len(self.metadata_df))
        
        # PDB code filter
        if self.valid_pdb_codes is not None:
            mask &= self.metadata_df['pdb_code'].isin(self.valid_pdb_codes)
        
        # Metal identity filter
        if self.valid_metals is not None:
            def has_valid_metal(metal_str):
                if pd.isna(metal_str) or not metal_str:
                    return False
                metals_in_site = set(metal_str.split(','))
                # Site must contain at least one of the valid metals
                return bool(metals_in_site & self.valid_metals)
            
            mask &= self.metadata_df['metal'].apply(has_valid_metal)
        
        # Numeric filters
        filters = [
            ('n_metals', self.min_metals, self.max_metals),
            ('n_organic_ligands', self.min_organic_ligands, self.max_organic_ligands),
            ('n_waters', self.min_waters, self.max_waters),
            ('n_nucleotides', self.min_nucleotides, self.max_nucleotides),
            ('n_amino_acids', self.min_amino_acids, self.max_amino_acids),
            ('n_coordinating_amino_acids', self.min_co, self.max_co),
        ]
        
        for col, min_val, max_val in filters:
            if min_val is not None:
                mask &= self.metadata_df[col] >= min_val
            if max_val is not None:
                mask &= self.metadata_df[col] <= max_val
        
        # Sites per PDB filter
        if self.min_sites_per_pdb is not None or self.max_sites_per_pdb is not None:
            sites_per_pdb = self.metadata_df.groupby('pdb_code').size()
            
            valid_pdbs = sites_per_pdb.index
            if self.min_sites_per_pdb is not None:
                valid_pdbs = valid_pdbs[sites_per_pdb >= self.min_sites_per_pdb]
            if self.max_sites_per_pdb is not None:
                valid_pdbs = valid_pdbs[sites_per_pdb <= self.max_sites_per_pdb]
            
            mask &= self.metadata_df['pdb_code'].isin(valid_pdbs)
        
        # Store valid indices
        self.valid_indices = self.metadata_df.index[mask].tolist()
        self.filtered_metadata = self.metadata_df.loc[mask].reset_index(drop=True)
        
        logger.info(f"After filtering: {len(self.valid_indices)} sites from "
                   f"{self.filtered_metadata['pdb_code'].nunique()} PDBs")
    
    def get_metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame for all valid sites after filtering."""
        return self.filtered_metadata.copy()
    
    def get_all_metadata(self) -> pd.DataFrame:
        """Get complete metadata DataFrame before filtering."""
        return self.metadata_df.copy()
    
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
        
        # Load site from pickle
        site_filename = f"{site_info['pdb_code']}_{site_info['site_idx']}.pkl"
        site_path = self.cache_folder / "sites" / site_filename
        
        with open(site_path, 'rb') as f:
            site_chain = pickle.load(f)
        
        site_id = f"{site_info['pdb_code']}_{site_info['site_idx']}"
        return site_id, site_chain
    
    def get_site_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific site by index."""
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        metadata_idx = self.valid_indices[idx]
        return self.metadata_df.iloc[metadata_idx].to_dict()