# metalsitenn/constants.py
'''
* Author: Evan Komp
* Created: 10/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
AA_ATOMS = {
    'ALA': set(['N', 'CA', 'C', 'O', 'CB']),
    'ARG': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']),
    'ASN': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2']),
    'ASP': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2']),
    'CYS': set(['N', 'CA', 'C', 'O', 'CB', 'SG']),
    'GLN': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2']),
    'GLU': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2']),
    'GLY': set(['N', 'CA', 'C', 'O']),
    'HIS': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2']),
    'ILE': set(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1']),
    'LEU': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2']),
    'LYS': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ']),
    'MET': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE']),
    'PHE': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']),
    'PRO': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD']),
    'SER': set(['N', 'CA', 'C', 'O', 'CB', 'OG']),
    'THR': set(['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2']),
    'TRP': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']),
    'TYR': set(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']),
    'VAL': set(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'])
}

# List of common metal ions
METAL_IONS = set(['LI', 'BE', 'NA', 'MG', 'AL', 'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'PO', 'FR', 'RA', 'AC', 'TH', 'PA', 'U', 'NP', 'PU', 'AM', 'CM', 'BK', 'CF', 'ES', 'FM', 'MD', 'NO', 'LR'])

# record types for protein structures
RECORD_TYPES = {'ATOM', 'HETATM'}

# all possible atoms
ATOMS = {'C', 'N', 'H', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I'}
# convert specific types of atoms in protein to generic types
# Eg. CA -> C, CB -> C, etc.
def AA_ATOMS_TO_GENERIC(atom_name: str) -> str:
    """Convert specific atom names to generic atomic symbols.
    
    Args:
        atom_name: Specific atom name from PDB/protein structure
        
    Returns:
        Generic atomic symbol
        
    Raises:
        KeyError: If atom name cannot be converted to generic type
    """
    # 1. Direct match to ATOMS or METAL_IONS
    if atom_name in ATOMS or atom_name in METAL_IONS:
        return atom_name
        
    # 2. First letter match
    first = atom_name[0].upper()
    if first in ATOMS:
        return first
        
    # 3. First two letters match 
    if len(atom_name) >= 2:
        first_two = atom_name[:2].upper()
        if first_two in ATOMS:
            return first_two
            
    raise KeyError(f"Cannot convert {atom_name} to generic atom type")
