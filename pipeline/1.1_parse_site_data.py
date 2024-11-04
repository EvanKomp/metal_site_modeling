# src/parse_site_data.py
'''
* Author: Evan Komp
* Created: 10/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

For each available sire, determine if it contains non-water, non-metal heteroatoms, metals, or incomplete residues.
'''
import os
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import pandas as pd
import json

from metalsitenn.constants import AA_ATOMS, METAL_IONS

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='./logs/parse_site_data.log')

def analyze_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # Check for non-water, non-metal heteroatoms
    heteroatoms = set()
    metals = set()
    for residue in structure.get_residues():
        if residue.id[0] != ' ':  # This checks if it's a heteroatom
            resname = residue.resname
            if resname != 'HOH' and resname not in METAL_IONS:
                heteroatoms.add(resname)
            elif resname in METAL_IONS:
                metals.add(resname)

    # Check for incomplete residues
    incomplete_residues = []
    for residue in structure.get_residues():
        if residue.resname in AA_ATOMS:
            expected_atoms = AA_ATOMS[residue.resname]
            present_atoms = set(atom.name for atom in residue)
            missing_atoms = expected_atoms - present_atoms
            if missing_atoms:
                incomplete_residues.append(f"{residue.resname}{residue.id[1]}")

    return bool(heteroatoms), list(metals), incomplete_residues

def process_pdb_files(pdb_files, output_csv, output_metrics_json):
    results = []
    for pdb_file in pdb_files:
        has_heteroatoms, metals, incomplete_residues = analyze_pdb(pdb_file)
        results.append({
            'File': os.path.basename(pdb_file),
            'hetlig': has_heteroatoms,
            'metals': ', '.join(metals) if metals else 'None',
            'incomplete_residues': ', '.join(incomplete_residues) if incomplete_residues else None
        })

    # Write results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # save metrics
    metrics = {
        'n_sites': len(pdb_files),
        'frac_hetlig': sum(r['hetlig'] for r in results) / len(results),
        'frac_broken_residues': sum(bool(r['incomplete_residues']) for r in results) / len(results)
    }
    with open(output_metrics_json, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    source_folder = os.path.join('data', 'mf_sites')
    pdb_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.pdb')]
    output_csv = os.path.join('data', 'site_labels.csv')
    output_metrics_json = os.path.join('data', 'metrics', 'site_label_metrics.json')
    process_pdb_files(pdb_files, output_csv, output_metrics_json)