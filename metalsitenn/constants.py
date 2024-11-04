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