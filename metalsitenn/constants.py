# metalsitenn/constants.py
'''
* Author: Evan Komp
* Created: 10/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os


COMMON_PROTEIN_ATOMS = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se'}
ALL_METALS = {
    # Alkali metals (Group 1)
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    
    # Alkaline earth metals (Group 2)
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    
    # Transition metals (Groups 3-12)
    # Period 4
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    # Period 5
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    # Period 6
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    # Period 7
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    
    # Post-transition metals (poor metals)
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'Nh', 'Fl', 'Mc', 'Lv', 'Te',
    
    # Lanthanides (f-block, Period 6)
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    
    # Actinides (f-block, Period 7)
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
}

# For backward compatibility, keep METALS as alias to ALL_METALS
METALS = ALL_METALS

# Subset of metals commonly found in biological systems
BIOLOGICAL_METALS = {
    # Essential trace metals
    'Fe', 'Cu', 'Zn', 'Mn', 'Co', 'Ni', 'Mo', 'W',
    # Alkali/alkaline earth metals in biology
    'Na', 'K', 'Mg', 'Ca',
    # Other metals sometimes found in biological contexts
    'Cd', 'Hg', 'Pb', 'V', 'Cr', 'La'
}
assert BIOLOGICAL_METALS.issubset(ALL_METALS), "BIOLOGICAL_METALS must be a subset of ALL_METALS"

# Metals commonly found in protein structures (PDB statistics)
PROTEIN_METALS = {
    'Zn', 'Ca', 'Mg', 'Fe', 'Na', 'K', 'Mn', 'Cu', 'Co', 'Ni', 'Mo', 'Cd', 'Hg'
}
assert PROTEIN_METALS.issubset(ALL_METALS), "PROTEIN_METALS must be a subset of ALL_METALS"
# other critical metals we ar einterested in
CRITICAL_METALS = {
    'Li', 'Ni', 'Co', 'Ga', 'Pt', 'Mg', 'Ir', 'Dy', 'Nd', 'Pr', 'Tb', 'Ti', 'U', 'Cu', 'Mn', 'Al', 'Te', 'Au', 'Ag'}
assert CRITICAL_METALS.issubset(ALL_METALS), "CRITICAL_METALS must be a subset of ALL_METALS"

RESNAME_3LETTER = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                      'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                      'THR', 'TRP', 'TYR', 'VAL']


# allmetal3d groups
am3d_ALKAILI_METALS = {'Na', 'K'}
am3d_MG = {'Mg'}
am3d_CA = {'Ca'}
am3d_ZN = {'Zn'}
am3d_NON_ZN_TM = {'Fe', 'Co', 'Ni', 'Cu', 'Mn'}
am3d_UNCOVERED_METAL = ALL_METALS - (am3d_ALKAILI_METALS | am3d_MG | am3d_CA | am3d_ZN | am3d_NON_ZN_TM)


# Load element mappings from elements.txt
# used in the CIF parser but these numbers DOI NOT map to the same
# tokens in the model
def _load_elements():
    """Load element index to symbol mappings from elements.txt file."""
    DIR = os.path.dirname(__file__)
    elements_file = os.path.join(DIR, 'placer_modules', 'data', 'elements.txt')
    
    i2e = {}  # index to element symbol
    e2i = {}  # element symbol to index
    
    with open(elements_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                symbol = parts[1]
                i2e[idx] = symbol
                e2i[symbol] = idx
    
    return i2e, e2i

# Element mappings
I2E, E2I = _load_elements()


