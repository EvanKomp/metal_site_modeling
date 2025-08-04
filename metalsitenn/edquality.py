# metalsitenn/edquality.py
'''
* Author: Evan Komp
* Created: 7/29/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

'''
* Author: Evan Komp
* Created: 7/29/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gzip
from dataclasses import dataclass
from joblib import Parallel, delayed
import multiprocessing

from metalsitenn.constants import ALL_METALS

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain as BioChain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class MetalQualityEntry:
    """Single metal quality entry with coordinates and RSZD data."""
    pdb_id: str
    metal_symbol: str
    coordinates: Tuple[float, float, float]
    rszd: float
    rscc: Optional[float] = None
    chain: Optional[str] = None
    residue_number: Optional[str] = None
    raw_data: Optional[Dict] = None


class EDQualityMapping:
    """
    Electron density quality mapping for metal sites using BioPython.
    
    Reads CSV with metal quality data, parses corresponding CIF files using BioPython
    to extract coordinates, and creates an efficient spatial lookup index for quality assessment.
    """
    
    def __init__(self, csv_path: str, cif_folder: str = None, cache_path: str = None, 
                 coordinate_tolerance: float = 0.5, n_jobs: int = -1):
        """
        Initialize EDQualityMapping.
        
        Args:
            csv_path: Path to CSV file with metal quality data
            cif_folder: Path to folder containing CIF files (optional if loading from cache)
            cache_path: Path to save/load precomputed mapping (optional)
            coordinate_tolerance: Distance tolerance for coordinate matching in Angstroms
            n_jobs: Number of parallel jobs for processing (-1 for all cores)
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required for EDQualityMapping. Install with: pip install biopython")
            
        self.csv_path = Path(csv_path)
        self.cif_folder = Path(cif_folder) if cif_folder else None
        self.cache_path = Path(cache_path) if cache_path else None
        self.coordinate_tolerance = coordinate_tolerance
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # Main data structures
        self.quality_entries: List[MetalQualityEntry] = []
        self.pdb_lookup: Dict[str, List[MetalQualityEntry]] = defaultdict(list)
        self.coordinate_index: Dict[str, List[Tuple[np.ndarray, MetalQualityEntry]]] = defaultdict(list)
        
        # Load or build the mapping
        if self.cache_path and self.cache_path.exists():
            self.load_from_cache()
        else:
            self.build_mapping()
            if self.cache_path:
                self.save_to_cache()
    
    def build_mapping(self):
        """Build the quality mapping by parsing CSV and extracting coordinates from CIF files."""
        logger.info(f"Building EDQualityMapping using BioPython with {self.n_jobs} parallel jobs...")
        
        if self.cif_folder is None:
            raise ValueError("cif_folder must be provided when building mapping from scratch")
        
        # Load and validate CSV
        df = self._load_and_validate_csv()
        
        # Group by PDB ID for efficient processing
        pdb_groups = list(df.groupby('PDB ID'))
        
        # Process PDB groups in parallel
        if self.n_jobs > 1:
            results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(self._process_pdb_group_worker)(pdb_id, group) 
                for pdb_id, group in tqdm(pdb_groups, desc="Processing PDB files")
            )
        else:
            results = []
            for pdb_id, group in tqdm(pdb_groups, desc="Processing PDB files"):
                results.append(self._process_pdb_group_worker(pdb_id, group))
        
        # Collect results from parallel processing
        for pdb_entries in results:
            if pdb_entries:
                for entry in pdb_entries:
                    self.quality_entries.append(entry)
                    self.pdb_lookup[entry.pdb_id].append(entry)
                    
                    # Add to coordinate index for spatial lookup
                    coord_array = np.array(entry.coordinates)
                    self.coordinate_index[entry.pdb_id].append((coord_array, entry))
        
        logger.info(f"Built mapping with {len(self.quality_entries)} metal quality entries")
    
    def _load_and_validate_csv(self) -> pd.DataFrame:
        """Load and validate the quality CSV file."""
        df = pd.read_csv(self.csv_path)
        
        required_columns = ['PDB ID', 'Metal', 'Chain', 'Number', 'RSZD']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        
        # Filter out entries with invalid RSZD values
        initial_count = len(df)
        df = df[~df['RSZD'].isin(['n/a', 'N/A'])]
        df = df.dropna(subset=['RSZD'])
        df['RSZD'] = pd.to_numeric(df['RSZD'], errors='coerce')
        df = df.dropna(subset=['RSZD'])
        
        logger.info(f"Loaded {len(df)} valid entries from {initial_count} total CSV rows")
        return df
    
    def _process_pdb_group_worker(self, pdb_id: str, group: pd.DataFrame) -> List[MetalQualityEntry]:
        """
        Worker function for parallel processing of a single PDB file using BioPython.
        
        Args:
            pdb_id: PDB identifier
            group: DataFrame rows for this PDB
            
        Returns:
            List of MetalQualityEntry objects for this PDB
        """
        pdb_id = pdb_id.upper()
        entries = []
        
        # Find CIF file
        cif_path = self._find_cif_file(pdb_id)
        if not cif_path:
            logger.warning(f"CIF file not found for PDB {pdb_id}")
            return entries
        
        try:
            # Parse CIF file with BioPython
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(cif_path))
            
            # Extract metal coordinates for this PDB - match by atom numbers first
            metal_coords_by_numbers = self._extract_metal_coordinates_by_numbers(structure)
            
            # Match CSV entries with coordinates using atom numbers
            for _, row in group.iterrows():
                entry = self._match_entry_by_numbers(row, metal_coords_by_numbers, pdb_id)
                if entry:
                    entries.append(entry)
                    
        except Exception as e:
            logger.error(f"Error processing {pdb_id}: {e}")
        
        return entries
    
    def _extract_metal_coordinates_by_numbers(self, structure: Structure) -> Dict[Tuple[str, str, str, str], Tuple[float, float, float]]:
        """
        Extract coordinates for all metal atoms using atom numbers for matching.
        
        Args:
            structure: BioPython Structure object
            
        Returns:
            Dictionary mapping (chain_id, res_number, metal_symbol, atom_name) to (x, y, z) coordinates
        """
        metal_coords = {}
        
        # Define common metal elements by atomic symbol
        metal_symbols = ALL_METALS
        # add the upper case versions of the symbols
        metal_symbols = set(symbol.upper() for symbol in metal_symbols)
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                
                for residue in chain:
                    res_number = str(residue.get_id()[1])  # Get sequence number
                    
                    for atom in residue:
                        element = atom.element.upper()
                        atom_name = atom.get_name()
                        
                        # Check if this is a metal atom
                        if element in metal_symbols:
                            coords = atom.get_coord()
                            key = (chain_id, res_number, element, atom_name)
                            metal_coords[key] = (float(coords[0]), float(coords[1]), float(coords[2]))
        
        return metal_coords
    
    def _match_entry_by_numbers(self, row: pd.Series, metal_coords: Dict, pdb_id: str) -> Optional[MetalQualityEntry]:
        """
        Match CSV entry with extracted coordinates using atom numbers and return entry if found.
        
        Args:
            row: CSV row data
            metal_coords: Dictionary of metal coordinates keyed by (chain, res_num, metal, atom_name)
            pdb_id: PDB identifier
            
        Returns:
            MetalQualityEntry if coordinates found, None otherwise
        """
        chain = str(row['Chain']).strip()
        res_num = str(row['Number']).strip()
        metal_symbol = str(row['Metal']).strip().upper()
        
        # Find matching coordinates by chain, residue number, and metal symbol
        matching_coords = None
        matched_atom_name = None
        
        for (c, r, m, a), coords in metal_coords.items():
            if c == chain and r == res_num and m == metal_symbol:
                matching_coords = coords
                matched_atom_name = a
                break
        
        # If exact chain match fails, try matching by residue number and metal only
        if matching_coords is None:
            for (c, r, m, a), coords in metal_coords.items():
                if r == res_num and m == metal_symbol:
                    matching_coords = coords
                    matched_atom_name = a
                    logger.debug(f"Matched {pdb_id} {res_num} {metal_symbol} with different chain: {c} vs {chain}")
                    break
        
        if matching_coords is not None:
            # Create quality entry
            return MetalQualityEntry(
                pdb_id=pdb_id,
                metal_symbol=metal_symbol,
                coordinates=matching_coords,
                rszd=float(row['RSZD']),
                rscc=row.get('RSCC') if pd.notna(row.get('RSCC')) else None,
                chain=chain,
                residue_number=res_num,
                raw_data=row.to_dict()
            )
        else:
            logger.debug(f"No coordinates found for {pdb_id} {chain}:{res_num} {metal_symbol}")
            return None
    
    def _find_cif_file(self, pdb_id: str) -> Optional[Path]:
        """Find CIF file for given PDB ID in the CIF folder."""
        possible_names = [
            f"{pdb_id.lower()}.cif",
            f"{pdb_id.upper()}.cif", 
            f"{pdb_id.lower()}.cif.gz",
            f"{pdb_id.upper()}.cif.gz"
        ]
        
        for name in possible_names:
            path = self.cif_folder / name
            if path.exists():
                return path
        
        # Check subdirectories (common in PDB mirror structures)
        middle_chars = pdb_id[1:3].lower()
        subdir = self.cif_folder / middle_chars
        if subdir.exists():
            for name in possible_names:
                path = subdir / name
                if path.exists():
                    return path
        
        return None
    
    def get_metal_quality(self, pdb_id: str, metal_symbol: str, coordinates: Tuple[float, float, float]) -> Optional[MetalQualityEntry]:
        """
        Get quality data for a metal atom by PDB ID, symbol, and coordinates.
        
        Args:
            pdb_id: PDB identifier
            metal_symbol: Metal element symbol
            coordinates: (x, y, z) coordinates
            
        Returns:
            MetalQualityEntry if found within tolerance, None otherwise
        """
        pdb_id = pdb_id.upper()
        if pdb_id not in self.coordinate_index:
            return None
        
        target_coord = np.array(coordinates)
        
        for coord_array, entry in self.coordinate_index[pdb_id]:
            # Check metal symbol and coordinate distance
            if (entry.metal_symbol == metal_symbol.upper() and 
                np.linalg.norm(coord_array - target_coord) <= self.coordinate_tolerance):
                return entry
        
        return None
    
    def assess_site_quality(self, pdb_id: str, metal_atoms: List, max_rszd_threshold: float) -> bool:
        """
        Assess if all metals in a site meet the quality threshold.
        
        Args:
            pdb_id: PDB identifier
            metal_atoms: List of metal atoms with .element and .xyz attributes
            max_rszd_threshold: Maximum RSZD value for quality metal
            
        Returns:
            True if ALL metals in site are quality (RSZD <= threshold), False otherwise
        """
        if not metal_atoms:
            return False
        
        for metal_atom in metal_atoms:
            # Handle different atom object types
            if hasattr(metal_atom, 'metal') and not metal_atom.metal:
                continue
            elif hasattr(metal_atom, 'element'):
                element = metal_atom.element
            else:
                continue
                
            if hasattr(metal_atom, 'xyz'):
                coords = (float(metal_atom.xyz[0]), float(metal_atom.xyz[1]), float(metal_atom.xyz[2]))
            elif hasattr(metal_atom, 'get_coord'):
                coord_array = metal_atom.get_coord()
                coords = (float(coord_array[0]), float(coord_array[1]), float(coord_array[2]))
            else:
                continue
                
            quality_entry = self.get_metal_quality(pdb_id, element, coords)
            
            if quality_entry is None:
                # No quality data available - consider as failed quality check
                return False
            
            if quality_entry.rszd > max_rszd_threshold:
                # Metal fails quality threshold
                return False
        
        # All metals passed quality checks
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the quality mapping."""
        if not self.quality_entries:
            return {"total_entries": 0}
        
        rszd_values = [entry.rszd for entry in self.quality_entries]
        metal_counts = defaultdict(int)
        pdb_counts = defaultdict(int)
        
        for entry in self.quality_entries:
            metal_counts[entry.metal_symbol] += 1
            pdb_counts[entry.pdb_id] += 1
        
        return {
            "total_entries": len(self.quality_entries),
            "unique_pdbs": len(self.pdb_lookup),
            "rszd_mean": np.mean(rszd_values),
            "rszd_std": np.std(rszd_values),
            "rszd_min": np.min(rszd_values),
            "rszd_max": np.max(rszd_values),
            "metal_counts": dict(metal_counts),
            "entries_per_pdb_mean": np.mean(list(pdb_counts.values())),
        }
    
    def save_to_cache(self):
        """Save the mapping to cache file."""
        if self.cache_path is None:
            return
        
        cache_data = {
            'quality_entries': self.quality_entries,
            'pdb_lookup': dict(self.pdb_lookup),
            'coordinate_index': {pdb: [(coord.tolist(), entry) for coord, entry in entries] 
                               for pdb, entries in self.coordinate_index.items()},
            'coordinate_tolerance': self.coordinate_tolerance,
            'n_jobs': self.n_jobs,
            'csv_path': str(self.csv_path)
        }
        
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        if str(self.cache_path).endswith('.gz'):
            with gzip.open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        logger.info(f"Saved quality mapping cache to {self.cache_path}")
    
    def load_from_cache(self):
        """Load the mapping from cache file."""
        logger.info(f"Loading quality mapping from cache: {self.cache_path}")
        
        if str(self.cache_path).endswith('.gz'):
            with gzip.open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
        else:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
        
        self.quality_entries = cache_data['quality_entries']
        self.pdb_lookup = defaultdict(list, cache_data['pdb_lookup'])
        
        # Reconstruct coordinate index with numpy arrays
        self.coordinate_index = defaultdict(list)
        for pdb, entries in cache_data['coordinate_index'].items():
            self.coordinate_index[pdb] = [(np.array(coord), entry) for coord, entry in entries]
        
        self.coordinate_tolerance = cache_data.get('coordinate_tolerance', 0.5)
        self.n_jobs = cache_data.get('n_jobs', multiprocessing.cpu_count())
        
        logger.info(f"Loaded {len(self.quality_entries)} quality entries from cache")