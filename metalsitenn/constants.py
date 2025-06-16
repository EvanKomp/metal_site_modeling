# metalsitenn/constants.py
'''
* Author: Evan Komp
* Created: 10/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os

# List of common metal ions
COMMON_PROTEIN_ATOMS = {'C', 'N', 'H', 'D', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se'}

# Load element mappings from elements.txt
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


