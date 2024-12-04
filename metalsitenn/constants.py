# metalsitenn/constants.py
'''
* Author: Evan Komp
* Created: 10/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

# List of common metal ions
METAL_IONS = set(['LI', 'BE', 'NA', 'MG', 'AL', 'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'PO', 'FR', 'RA', 'AC', 'TH', 'PA', 'U', 'NP', 'PU', 'AM', 'CM', 'BK', 'CF', 'ES', 'FM', 'MD', 'NO', 'LR'])
COMMON_PROTEIN_ATOMS = {'C', 'N', 'H', 'D', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I', 'SE'}
UNCOMMON_PROTEIN_ATOMS = {'AS', 'B'}

# record types for protein structures
RECORD_TYPES = {'ATOM', 'HETATM'}

# all possible atoms we would see in proteins besides metal ions
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
