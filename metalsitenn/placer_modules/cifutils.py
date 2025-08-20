# metalsitenn/placer_modules/citfutils.py
'''
See original code at : https://github.com/baker-laboratory/PLACER/blob/main/modules/cifutils.py

Modifications:
- Method to extract subset of atoms associated with a metal binding site.
- saveing is now more complete, with atom names
'''

import sys,os
import json
import gzip
import re
import copy
import collections
import random
from pathlib import Path
from openbabel import openbabel
import itertools
from typing import Dict,List, Union, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import torch

import pdbx
# print(pdbx.__dir__())
# import pdbx.reader as reader
# print(reader.__dir__())
# import pdbx.reader.PdbxReader
# import reader.PdbxReader as PdbxReader
# from PdbxReader import PdbxReader
from pdbx.reader.PdbxReader import PdbxReader
from metalsitenn.placer_modules import obutils
from metalsitenn.constants import RESNAME_3LETTER
from metalsitenn.constants import I2E

import logging
logger = logging.getLogger(__name__)

# ============================================================
Atom = collections.namedtuple('Atom', [
    'name',
    'xyz', # Cartesian coordinates of the atom
    'occ', # occupancy
    'bfac', # B-factor
    'leaving', # boolean flag to indicate whether the atom leaves the molecule upon bond formation
    'leaving_group', # list of atoms which leave the molecule if a bond with this atom is formed
    'parent', # neighboring heavy atom this atom is bonded to
    'element', # atomic number (1..118)
    'metal', # is this atom a metal? (bool)
    'charge', # atomic charge (int)
    'hyb', # hybridization state (int)
    'nhyd', # number of hydrogens
    'hvydeg', # heavy atom degree
    'align', # atom name alignment offset in PDB atom field
    'hetero'
])

Bond = collections.namedtuple('Bond', [
    'a','b', # names of atoms forming the bond (str)
    'aromatic', # is the bond aromatic? (bool)
    'in_ring', # is the bond in a ring? (bool)
    'order', # bond order (int)
    'intra', # is the bond intra-residue? (bool)
    'length' # reference bond length from openbabel (float)
])

Residue = collections.namedtuple('Residue', [
    'name',
    'atoms',
    'bonds',
    'automorphisms',
    'chirals',
    'planars',
    'alternatives'
])

Chain = collections.namedtuple('Chain', [
    'id',           # chain identifier
    'type',         # chain type (e.g. 'polypeptide(L)')
    'sequence',     # canonical sequence string
    'residues',     # Dict[str, Residue] - residue objects by position
    'atoms',        # Dict[tuple, Atom] - all atoms keyed by (chain_id, res_num, res_name, atom_name)
    'bonds',        # List[Bond] - all bonds including inter-residue
    'chirals',      # List[List[tuple]] - chiral center constraints
    'planars',      # List[List[tuple]] - planar constraint groups
    'automorphisms' # List[List[List[tuple]]] - symmetry groups
])


# ============================================================
def ParsePDBLigand(cifname : str) -> Dict:
    '''Parse a single molecule from the PDB-Ligands set
    '''

    data = []
    with open(cifname,'r') as cif:
        reader = PdbxReader(cif)
        reader.read(data)
    data = data[0]
    chem_comp_atom = data.getObj('chem_comp_atom')
    rows = chem_comp_atom.getRowList()

    # parse atom names
    idx = chem_comp_atom.getIndex('atom_id')
    atom_id = np.array([r[idx] for r in rows])
    
    # parse element symbols
    idx = chem_comp_atom.getIndex('type_symbol')
    symbol = np.array([r[idx] for r in rows])

    # parse leaving flags
    idx = chem_comp_atom.getIndex('pdbx_leaving_atom_flag')
    leaving = [r[idx] for r in rows]
    leaving = np.array([True if flag=='Y' else False for flag in leaving], dtype=bool)

    # atom name alignment offset in PDB atom field
    idx = chem_comp_atom.getIndex('pdbx_align')
    pdbx_align = np.array([int(r[idx]) for r in rows])
    
    # parse xyz coordinates
    i = chem_comp_atom.getIndex('model_Cartn_x')
    j = chem_comp_atom.getIndex('model_Cartn_y')
    k = chem_comp_atom.getIndex('model_Cartn_z')
    xyz = [(r[i],r[j],r[k]) for r in rows]
    xyz = np.array([[float(c) if c!='?' else np.nan for c in p] for p in xyz])

    out = {'atom_id' : atom_id,
           'leaving' : leaving,
           'symbol' : symbol,
           'pdbx_align' : pdbx_align,
           'xyz' : xyz}

    return out


# ============================================================
class CIFParser:
    
    def __init__(self, skip_res : List[str] = None, mols=None):
        
        # parse pre-compiled library of all residues observed in the PDB
        DIR = os.path.dirname(__file__)
        if mols is None:
            with gzip.open(f'{DIR}/data/ligands.json.gz','rt') as file:
                self.mols = json.load(file)
        else:
            # IK: added an option for users to provide an edited library if they want to add new non-canonicals
            self.mols = mols

        # skip-residues are deleted form the library
        if skip_res is not None:
            for res in skip_res:
                if res in self.mols.keys():
                    del self.mols[res]

        # parse the quasi-symmetric groups table
        df = pd.read_csv(f'{DIR}/data/quasisym.csv')
        df.indices = df.indices.apply(lambda x : [int(xi) for xi in x.split(',')])
        df['matcher'] = df.apply(lambda x : openbabel.OBSmartsPattern(), axis=1)
        df.apply(lambda x : x.matcher.Init(x.smarts), axis=1)
        self.quasisym = {smarts:(matcher,torch.tensor(indices))
                         for smarts,matcher,indices 
                         in zip(df.smarts,df.matcher,df.indices)}
        
        # parse periodic table
        with open(f'{DIR}/data/elements.txt','r') as f:
            self.i2a = [l.strip().split()[:2] for l in f.readlines()]
            self.i2a = {int(i):a for i,a in self.i2a}
    

    def getRes(self,resname : str) -> Residue:
        '''get a residue from the library; residues are loaded dynamically'''
        
        res = self.mols.get(resname)
        if res is None:
            logger.debug(f'Residue {resname} not found in the library, not parsed.')
            return res
        
        if 'res' not in res.keys():
            res['res'] = self.parseLigand(sdfstring=res['sdf'],
                                          atom_id=res['atom_id'],
                                          leaving=res['leaving'],
                                          pdbx_align=res['pdbx_align'])
        return res
        
        
    def GetEquibBondLength(self, 
                           a: Atom,
                           b: Atom,
                           order : int = 1,
                           aromatic : bool = False) -> float:
        '''find equilibrium bond length between two atoms
        Adapted from: https://github.com/openbabel/openbabel/blob/master/src/bond.cpp#L575
        '''
        
        def CorrectedBondRad(elem, hyb):
            '''Return a "corrected" bonding radius based on the hybridization.
            Scale the covalent radius by 0.95 for sp2 and 0.90 for sp hybridsation
            '''
            rad = openbabel.GetCovalentRad(elem)
            if hyb==2:
                return rad * 0.95
            elif hyb==1:
                return rad * 0.90
            else:
                return rad
        
        rad_a = CorrectedBondRad(a.element, a.hyb)
        rad_b = CorrectedBondRad(b.element, b.hyb)
        length = rad_a + rad_b

        if aromatic==True:
            return length * 0.93

        if order==3:
            return length * 0.87
        elif order==2:
            return length * 0.91
        
        return length

    
    def AddQuasisymmetries(self, 
                           obmol : openbabel.OBMol,
                           automorphisms : torch.Tensor) -> torch.Tensor:
        '''add quasisymmetries to automorphisms
        '''

        renum = []
        for smarts,(matcher,indices) in self.quasisym.items():
            res = openbabel.vectorvInt()
            if matcher.Match(obmol,res,0):
                res = torch.tensor(res)[:,indices]-1
                res = res.sort(-1)[0]
                res = torch.unique(res,dim=0)
                for res_i in res:
                    res_i = torch.tensor(list(itertools.permutations(res_i,indices.shape[0])))
                    renum.append(res_i)
                
        if len(renum)<1:
            return automorphisms
        elif len(renum)==1:
            renum = renum[0]
        else:
            random.shuffle(renum)
            renum = renum[:4]
            renum = torch.stack([torch.cat(ijk) for ijk in itertools.product(*renum)])

        L = automorphisms.shape[-1]
        modified = automorphisms[:,None].repeat(1,renum.shape[0],1)
        modified[...,renum[0]]=automorphisms[:,renum]
        modified = modified.reshape(-1,L)
        modified = torch.unique(modified, dim=0)
        
        return modified


    @staticmethod
    def getLeavingAtoms(a,leaving,s):
        for b in openbabel.OBAtomAtomIter(a):
            if leaving[b.GetIndex()]==True:
                if b.GetIndex() not in s:
                    s.append(b.GetIndex())
                    CIFParser.getLeavingAtoms(b,leaving,s)


    @staticmethod
    def getLeavingAtoms2(aname, G):

        leaving_group = set()
    
        if G.nodes[aname]['leaving']==True:
            return []

        for m in G.neighbors(aname):
            if G.nodes[m]['leaving']==False:
                continue
            leaving_group.update({m})
            H = G.subgraph(set(G.nodes)-{m})
            ccs = list(nx.connected_components(H))
            if len(ccs)>1:
                for cc in ccs:
                    if aname not in cc:
                        leaving_group.update(cc)

        return list(leaving_group)


    #@staticmethod
    def parseLigand(self,
                    sdfstring : str,
                    atom_id : List[str],
                    leaving : List[bool],
                    pdbx_align : List[int],) -> Residue:

        # create molecule from the sdf string
        obmol = openbabel.OBMol()
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("sdf")
        obConversion.ReadString(obmol,sdfstring)
        
        # correct for pH to get some charged groups
        obmol_ph = openbabel.OBMol(obmol)
        obmol_ph.CorrectForPH()
        obmol_ph.DeleteHydrogens()
        ha_iter = openbabel.OBMolAtomIter(obmol_ph)                
        
        # get atoms and their features
        atoms = {}
        for aname,aleaving,align,a in zip(atom_id,leaving,pdbx_align,openbabel.OBMolAtomIter(obmol)):

            # parent heavy atoms
            parent = None
            for b in openbabel.OBAtomAtomIter(a):
                if b.GetAtomicNum()>1:
                    parent = atom_id[b.GetIndex()]
            
            charge = a.GetFormalCharge()
            nhyd = a.ExplicitHydrogenCount()
            if a.GetAtomicNum()>1:
                ha = next(ha_iter)
                charge = ha.GetFormalCharge()
                nhyd = ha.GetTotalDegree()-ha.GetHvyDegree()
            
            atoms[aname] = Atom(name=aname,
                                xyz=[0.0,0.0,0.0],
                                occ=0.0,
                                bfac=0.0,
                                leaving=aleaving,
                                leaving_group=[],
                                parent=parent,
                                element=a.GetAtomicNum(),
                                metal=a.IsMetal(),
                                charge=charge,
                                hyb=a.GetHyb(),
                                nhyd=nhyd,
                                align=align,
                                hvydeg=a.GetHvyDegree(),
                                hetero=False)
        
        # get bonds and their features
        bonds = []
        for b in openbabel.OBMolBondIter(obmol):
            bonds.append(Bond(a=atom_id[b.GetBeginAtom().GetIndex()],
                              b=atom_id[b.GetEndAtom().GetIndex()],
                              aromatic=b.IsAromatic(),
                              in_ring=b.IsInRing(),
                              order=b.GetBondOrder(),
                              intra=True,
                              length=b.GetLength()))

        # get automorphisms
        automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
        
        # add quasi-symmetric groups
        automorphisms = self.AddQuasisymmetries(obmol, automorphisms)
        
        # only retain atoms with alternative mappings
        mask = (automorphisms[:1]==automorphisms).all(dim=0)
        automorphisms = automorphisms[:,~mask]

        # skip automorphisms which include leaving atoms
        if automorphisms.shape[0]>1:
            mask = torch.tensor(leaving)[automorphisms].any(dim=0)
            automorphisms = automorphisms[:,~mask]
            if automorphisms.shape[-1]>0:
                automorphisms = torch.unique(automorphisms,dim=0)
            else:
                automorphisms = automorphisms.flatten()

        # get chirals and planars
        chirals = obutils.GetChirals(obmol, heavy=True)
        planars = obutils.GetPlanars(obmol, heavy=True)

        # add leaving groups to atoms
        G = nx.Graph()
        G.add_nodes_from([(a.name,{'leaving':a.leaving}) for a in atoms.values()])
        G.add_edges_from([(bond.a,bond.b) for bond in bonds])
        for k,v in atoms.items():
            atoms[k] = v._replace(leaving_group=CIFParser.getLeavingAtoms2(k,G))
        
        # put everything into a residue
        anames = np.array(atom_id)
        R = Residue(name=obmol.GetTitle(),
                    atoms=atoms,
                    bonds=bonds,
                    automorphisms=anames[automorphisms].tolist(),
                    chirals=anames[chirals].tolist(),
                    planars=anames[planars].tolist(),
                    alternatives=set())

        return R

    
    @staticmethod
    def parseOperationExpression(expression : str) -> List:
        '''a function to parse _pdbx_struct_assembly_gen.oper_expression 
        into individual operations'''

        expression = expression.strip('() ')
        operations = []
        for e in expression.split(','):
            e = e.strip()
            pos = e.find('-')
            if pos>0:
                start = int(e[0:pos])
                stop = int(e[pos+1:])
                operations.extend([str(i) for i in range(start,stop+1)])
            else:
                operations.append(e)

        return operations


    @staticmethod
    def parseAssemblies(data : pdbx.reader.PdbxContainers.DataContainer) -> Dict:
        '''parse biological assembly data'''
        
        assembly_data = data.getObj("pdbx_struct_assembly")
        assembly_gen = data.getObj("pdbx_struct_assembly_gen")
        oper_list = data.getObj("pdbx_struct_oper_list")

        if (assembly_data is None) or (assembly_gen is None) or (oper_list is None):
            return {}

        # save all basic transformations in a dictionary
        opers = {}
        for k in range(oper_list.getRowCount()):
            key = oper_list.getValue("id", k)
            val = np.eye(4)
            for i in range(3):
                val[i,3] = float(oper_list.getValue("vector[%d]"%(i+1), k))
                for j in range(3):
                    val[i,j] = float(oper_list.getValue("matrix[%d][%d]"%(i+1,j+1), k))
            opers.update({key:val})

        chains,ids = [],[]
        xforms = []

        for index in range(assembly_gen.getRowCount()):

            # Retrieve the assembly_id attribute value for this assembly
            assemblyId = assembly_gen.getValue("assembly_id", index)
            ids.append(assemblyId)

            # Retrieve the operation expression for this assembly from the oper_expression attribute	
            oper_expression = assembly_gen.getValue("oper_expression", index)

            oper_list = [CIFParser.parseOperationExpression(expression) 
                         for expression in re.split('\(|\)', oper_expression) if expression]

            # chain IDs which the transform should be applied to
            chains.append(assembly_gen.getValue("asym_id_list", index).split(','))

            if len(oper_list)==1:
                xforms.append(np.stack([opers[o] for o in oper_list[0]]))
            elif len(oper_list)==2:
                xforms.append(np.stack([opers[o1]@opers[o2] 
                                        for o1 in oper_list[0] 
                                        for o2 in oper_list[1]]))
            else:
                print('Error in processing assembly')           
                return xforms

        # return xforms as a dict {asmb_id:[(chain_id,xform[4,4])]}
        out = {i:[] for i in set(ids)}
        for key,c,x in zip(ids,chains,xforms):
            out[key].extend(itertools.product(c,x))
            
        return out


    def parse(self, filename : str) -> Dict:
        
        ########################################################
        # 0. read a .cif file
        ########################################################
        data = []
        if filename.endswith('.gz'):
            with gzip.open(filename,'rt',encoding='utf-8') as cif:
                reader = PdbxReader(cif)
                reader.read(data)
        else:
            with open(filename,'r') as cif:
                reader = PdbxReader(cif)
                reader.read(data)
        data = data[0]
        filepath = Path(filename)
        file_name_no_ext = filepath.stem


        ########################################################
        # 1. parse mappings of modified residues to their 
        #    standard counterparts
        ########################################################
        pdbx_struct_mod_residue = data.getObj('pdbx_struct_mod_residue')
        if pdbx_struct_mod_residue is None:
            modres = {}
        else:
            modres = {(r[pdbx_struct_mod_residue.getIndex('label_comp_id')],
                       r[pdbx_struct_mod_residue.getIndex('parent_comp_id')])
                      for r in pdbx_struct_mod_residue.getRowList()}
            modres = {k:v for k,v in modres if k!=v}


        ########################################################
        # 2. parse polymeric chains
        ########################################################
        pdbx_poly_seq_scheme = data.getObj('pdbx_poly_seq_scheme')
        chains = {}
        if pdbx_poly_seq_scheme is not None:
            
            # establish mapping asym_id <--> (entity_id,pdb_strand_id)
            chains = {
                r[pdbx_poly_seq_scheme.getIndex('asym_id')]: {
                    'entity_id' : r[pdbx_poly_seq_scheme.getIndex('entity_id')],
                    'pdb_strand_id' : r[pdbx_poly_seq_scheme.getIndex('pdb_strand_id')]}
                for r in pdbx_poly_seq_scheme.getRowList() }
            
            # parse canonical 1-letter sequences
            entity_poly = data.getObj('entity_poly')
            if entity_poly is not None:
                for r in entity_poly.getRowList():
                    entity_id = r[entity_poly.getIndex('entity_id')]
                    type_ = r[entity_poly.getIndex('type')]
                    seq = r[entity_poly.getIndex('pdbx_seq_one_letter_code_can')].replace('\n','')
                    for k,v in chains.items():
                        if v['entity_id']==entity_id:
                            v['type'] = r[entity_poly.getIndex('type')]
                            v['seq'] = seq

            # parse residues that are actually present in the polymer
            entity_poly_seq = data.getObj('entity_poly_seq')
            residues = [(r[entity_poly_seq.getIndex('entity_id')],
                         r[entity_poly_seq.getIndex('num')],
                         r[entity_poly_seq.getIndex('mon_id')],
                         r[entity_poly_seq.getIndex('hetero')] in {'y','yes'}) 
                        for r in entity_poly_seq.getRowList()]
            for entity_id,res in itertools.groupby(residues, key=lambda x : x[0]):
                res = [resi[1:] for resi in list(res) if resi[2] in self.mols.keys()]
                
                # when there are alternative residues at the same position
                # pick the one which occurs first
                res = {k:Residue(*self.getRes(next(v)[1])['res'][:-1],alternatives=set([vi[1] for vi in v]))
                       for k,v in itertools.groupby(res, key=lambda x : x[0])}

                for k,v in chains.items():
                    if v['entity_id']==entity_id:
                        v['res'] = {k:copy.deepcopy(v) for k,v in res.items()}


        ########################################################
        # 3. parse non-polymeric molecules
        ########################################################
        
        # parse from HETATM
        atom_site = data.getObj('atom_site')
        comp_id_key = "auth_comp_id"
        if not atom_site.hasAttribute(comp_id_key):
            comp_id_key = "label_comp_id"  # alternative column label for residue name if `auth_comp_id` is missing
        assert atom_site.hasAttribute(comp_id_key), "Input CIF structure is missing a key for `comp_id`"

        nonpoly_res = [(r[atom_site.getIndex('label_asym_id')],
                        r[atom_site.getIndex('label_entity_id')],
                        r[atom_site.getIndex('auth_asym_id')],
                        r[atom_site.getIndex('auth_seq_id')], # !!! this is not necessarily an integer number, per mmcif specifications !!!
                        r[atom_site.getIndex(comp_id_key)]) 
                       for r in atom_site.getRowList() 
                       if r[atom_site.getIndex('group_PDB')]=='HETATM' and r[atom_site.getIndex('label_asym_id')] not in chains.keys()]
        nonpoly_res = [r for r in nonpoly_res if r[0] not in chains.keys()]
        nonpoly_chains = {r[0]:{'entity_id':r[1], 'pdb_strand_id':r[2],'type':'nonpoly','res':{}} for r in nonpoly_res}
        for r in nonpoly_res:
            #res = self.mols.get(r[4])
            res = self.getRes(r[4])
            if res is not None:
                res = res['res']
            nonpoly_chains[r[0]]['res'][r[3]] = res
        for v in nonpoly_chains.values():
            v['res'] = {k2:copy.deepcopy(v2) for k2,v2 in v['res'].items()}
        chains.update(nonpoly_chains)


        ########################################################
        # 4. populate residues with coordinates & track resolved residues
        ########################################################
        
        # Track which residues actually appear in atom_site records # unresolved filtering
        resolved_residues = set()  # unresolved filtering
        
        i = {k:atom_site.getIndex(val) for k,val in [('hetero', 'group_PDB'),
                                                    ('symbol', 'type_symbol'),
                                                    ('atm', 'label_atom_id'), # atom name
                                                    ('res', 'label_comp_id'), # residue name (3-letter)
                                                    ('chid', 'label_asym_id'), # chain ID
                                                    ('num', 'label_seq_id'), # sequence number
                                                    ('num_author', 'auth_seq_id'), # sequence number assigned by the author
                                                    ('alt', 'label_alt_id'), # alternative location ID
                                                    ('x', 'Cartn_x'), # xyz coords
                                                    ('y', 'Cartn_y'),
                                                    ('z', 'Cartn_z'),
                                                    ('occ', 'occupancy'), # occupancy
                                                    ('bfac', 'B_iso_or_equiv'), # B-factors 
                                                    ('model', 'pdbx_PDB_model_num') # model number (for multi-model PDBs, e.g. NMR)
                                                    ]}

        for r in atom_site.getRowList():

            hetero, symbol, atm, res, chid, num, num_author, alt, x, y, z, occ, bfac, model = \
                (t(r[i[k]]) for k,t in (('hetero',str), ('symbol',str), ('atm',str), ('res',str), ('chid',str), 
                                        ('num', str), ('num_author',str), ('alt',str),
                                        ('x',float), ('y',float), ('z',float), 
                                        ('occ',float), ('bfac',float), ('model',int)))
            
            # we use author assigned residue numbers for non-polymeric chains
            if chains[chid]['type']=='nonpoly':
                num = num_author
            if num=='.': # !!! fixes 1ZY8 is which FAD ligand is assigned to a polypeptide chain O !!!
                continue
            if num not in chains[chid]['res'].keys():
                continue
            
            # Track this residue as resolved since it appears in atom_site # unresolved filtering
            resolved_residues.add((chid, num))  # unresolved filtering
            
            residue = chains[chid]['res'][num]
            # skip if residue is not in the library
            if residue is not None and residue.name==res:
                # if any heavy atom in a residue cannot be matched
                # then mask the whole residue
                if atm not in residue.atoms.keys():
                    if symbol!='H' and symbol!='D':
                        chains[chid]['res'][num] = None
                    continue
                atom = residue.atoms[atm]
                if occ>atom.occ:
                    residue.atoms[atm] = atom._replace(xyz=[x,y,z], 
                                                    occ=occ,
                                                    bfac=bfac,
                                                    hetero=(hetero=='HETATM'))


        ########################################################
        # Filter out unresolved residues
        ########################################################
        
        # Remove residues that didn't appear in atom_site records # unresolved filtering
        for chain_id, chain_data in chains.items():  # unresolved filtering
            if 'res' in chain_data:  # unresolved filtering
                original_res_count = len(chain_data['res'])  # unresolved filtering
                # Keep only residues that were resolved in atom_site # unresolved filtering
                chain_data['res'] = {res_num: residue for res_num, residue in chain_data['res'].items()  # unresolved filtering
                                if (chain_id, res_num) in resolved_residues}  # unresolved filtering
                filtered_res_count = len(chain_data['res'])  # unresolved filtering
                if original_res_count != filtered_res_count:  # unresolved filtering
                    logger.debug(f"Chain {chain_id}: filtered {original_res_count - filtered_res_count} unresolved residues")  # unresolved filtering


        ########################################################
        # 5. parse covalent connections
        ########################################################
        
        struct_conn = data.getObj('struct_conn')
        if struct_conn is not None:
            covale = [(r[struct_conn.getIndex('ptnr1_label_asym_id')],
                       r[struct_conn.getIndex('ptnr1_label_seq_id')],
                       r[struct_conn.getIndex('ptnr1_auth_seq_id')],
                       r[struct_conn.getIndex('ptnr1_label_comp_id')],
                       r[struct_conn.getIndex('ptnr1_label_atom_id')],
                       r[struct_conn.getIndex('ptnr2_label_asym_id')],
                       r[struct_conn.getIndex('ptnr2_label_seq_id')],
                       r[struct_conn.getIndex('ptnr2_auth_seq_id')],
                       r[struct_conn.getIndex('ptnr2_label_comp_id')],
                       r[struct_conn.getIndex('ptnr2_label_atom_id')])
                      for r in struct_conn.getRowList() if r[struct_conn.getIndex('conn_type_id')]=='covale']
            F = lambda x : x[2] if chains[x[0]]['type']=='nonpoly' else x[1]
            # here we skip intra-residue covalent bonds assuming that
            # they are properly handled by parsing from the residue library
            covale = [((c[0],F(c[:4]),c[3],c[4]),(c[5],F(c[5:]),c[8],c[9])) 
                      for c in covale if c[:4]!=c[5:8]]

        else:
            covale = []


        ########################################################
        # 6. build connected chains
        ########################################################
        return_chains = {}
        for chain_id,chain in chains.items():
                        
            residues = list(chain['res'].items())
            atoms,bonds,skip_atoms = [],[],[]
            
            # (a) add inter-residue connections in polymers
            if 'polypept' in chain['type']:
                ab = ('C','N')
            elif 'polyribo' in chain['type'] or 'polydeoxyribo' in chain['type']:
                ab = ("O3'",'P')
            else:
                ab = ()

            if len(ab)>0:
                for ra,rb in zip(residues[:-1],residues[1:]):
                    # check for skipped residues (the ones failed in step 4)
                    if ra[1] is None or rb[1] is None:
                        continue
                    a = ra[1].atoms.get(ab[0])
                    b = rb[1].atoms.get(ab[1])
                    if a is not None and b is not None:
                        bonds.append(Bond(
                            a=(chain_id,ra[0],ra[1].name,a.name),
                            b=(chain_id,rb[0],rb[1].name,b.name),
                            aromatic=False,
                            in_ring=False,
                            order=1, # !!! we assume that all inter-residue bonds are single !!!
                            intra=False,
                            length=self.GetEquibBondLength(a,b)
                        ))
                        skip_atoms.extend([(chain_id,ra[0],ra[1].name,ai) for ai in a.leaving_group])
                        skip_atoms.extend([(chain_id,rb[0],rb[1].name,bi) for bi in b.leaving_group])

            # (b) add connections parsed from mmcif's struct_conn record
            for ra,rb in covale:
                a = b = None
                if ra[0]==chain_id and ra[1] in chain['res'].keys() and chain['res'][ra[1]] is not None and chain['res'][ra[1]].name==ra[2]:
                    a = chain['res'][ra[1]].atoms[ra[3]]
                    skip_atoms.extend([(chain_id,*ra[1:3],ai) for ai in a.leaving_group])
                if rb[0]==chain_id and rb[1] in chain['res'].keys() and chain['res'][rb[1]] is not None and chain['res'][rb[1]].name==rb[2]:
                    b = chain['res'][rb[1]].atoms[rb[3]]
                    skip_atoms.extend([(chain_id,*rb[1:3],bi) for bi in b.leaving_group])
                if a is not None and b is not None:
                    bonds.append(Bond(
                        a=(chain_id,*ra[1:3],a.name),
                        b=(chain_id,*rb[1:3],b.name),
                        aromatic=False,
                        in_ring=False,
                        order=1, # !!! we assume that all inter-residue bonds are single !!!
                        intra=False,
                        length=self.GetEquibBondLength(a,b)
                    ))
                    
            # (c) collect atoms
            skip_atoms = set(skip_atoms)
            atoms = {(chain_id,r[0],r[1].name,aname):a for r in residues if r[1] is not None
                     for aname,a in r[1].atoms.items()}
            atoms = {aname:a._replace(name=aname) for aname,a in atoms.items() if aname not in skip_atoms}

            # (d) collect intra-residue bonds
            bonds_intra = [bond._replace(a=(chain_id,r[0],r[1].name,bond.a),
                                         b=(chain_id,r[0],r[1].name,bond.b))
                           for r in residues if r[1] is not None
                           for bond in r[1].bonds]
            bonds_intra = [bond for bond in bonds_intra 
                           if bond.a not in skip_atoms and \
                           bond.b not in skip_atoms]

            bonds.extend(bonds_intra)
            
            # (e) double check whether bonded atoms actually exist:
            #     some could be part of the skip_atoms set and thus removed
            bonds = [bond for bond in bonds if bond.a in atoms.keys() and bond.b in atoms.keys()]
            bonds = list(set(bonds))
            
            # (f) relabel chirals, planars and automorphisms 
            #     to include residue indices and names
            chirals = [[(chain_id,r[0],r[1].name,c) for c in chiral] 
                       for r in residues if r[1] is not None for chiral in r[1].chirals]
            
            planars = [[(chain_id,r[0],r[1].name,c) for c in planar] 
                       for r in residues if r[1] is not None for planar in r[1].planars]
            
            automorphisms = [[[(chain_id,r[0],r[1].name,a) 
                               for a in auto] for auto in r[1].automorphisms] 
                             for r in residues if r[1] is not None and len(r[1].automorphisms)>1]

            chirals = [c for c in chirals if all([ci in atoms.keys() for ci in c])]
            planars = [c for c in planars if all([ci in atoms.keys() for ci in c])]

            if len(atoms)>0:
                return_chains[chain_id] = Chain(id=chain_id,
                                                type=chain['type'],
                                                sequence=chain.get('seq'),
                                                residues=chain.get('res', {}),
                                                atoms=atoms,
                                                bonds=bonds,
                                                chirals=chirals,
                                                planars=planars,
                                                automorphisms=automorphisms)

                
        ########################################################
        # 6. parse assemblies
        ########################################################
        asmb = self.parseAssemblies(data)
        asmb = {k:[vi for vi in v if vi[0] in return_chains.keys()]
                for k,v in asmb.items()}

        
        # convert covalent links to Bonds
        covale = [Bond(a=c[0],
                       b=c[1],
                       aromatic=False,
                       in_ring=False,
                       order=1,
                       intra=False,
                       length=1.5)
                  for c in covale if c[0][0]!=c[1][0]]

        # make sure covale atoms exist;
        # reset bond length to equilibrium
        def get_bond_length(a,b):
            return self.GetEquibBondLength(
                a=return_chains[a[0]].atoms[a],
                b=return_chains[b[0]].atoms[b])
        
        covale = [c._replace(length=get_bond_length(c.a,c.b)) \
                  for c in covale if \
                  c.a[0] in return_chains.keys() and \
                  c.b[0] in return_chains.keys() and \
                  c.a in return_chains[c.a[0]].atoms.keys() and \
                  c.b in return_chains[c.b[0]].atoms.keys()]

        
        # fix charges and hydrogen counts for cases when
        # charged a atom is connected by an inter-residue bond
        bonds = [v.bonds for k,v in return_chains.items()] + [covale]
        for bond in itertools.chain(*bonds):
            if bond.intra==False:
                #'''
                for i in (bond.a,bond.b):
                    a = return_chains[i[0]].atoms[i]
                    
                    if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3 and a.hvydeg==1: # -NH3+
                        return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=1)
                    if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==2 and a.hvydeg==2: # -(NH2+)-
                        return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=0)
                    elif a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3 and a.hvydeg==0: # free NH3+ group
                        return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=2)
                    elif a.element==8 and a.charge==-1 and a.hyb==3 and a.nhyd==0:
                        return_chains[i[0]].atoms[i] = a._replace(charge=0)
                    elif a.element==8 and a.charge==-1 and a.hyb==2 and a.nhyd==0: # O-linked connections
                        return_chains[i[0]].atoms[i] = a._replace(charge=0)
                    elif a.charge!=0:
                        pass
                #'''

        res = None
        if data.getObj('refine') is not None:
            try:
                res = float(data.getObj('refine').getValue('ls_d_res_high',0))
            except:
                res = None
        if (data.getObj('em_3d_reconstruction') is not None) and (res is None):
            try:
                res = float(data.getObj('em_3d_reconstruction').getValue('resolution',0))
            except:
                res = None
        try:
            meta = {
                'method' : data.getObj('exptl').getValue('method',0).replace(' ','_'),
                'date' : data.getObj('pdbx_database_status').getValue('recvd_initial_deposition_date',0),
                'resolution' : res,
                'id': file_name_no_ext,
            }
        except AttributeError:
            meta = None

        return return_chains,asmb,covale,meta
    
    
    #@staticmethod
    def save(self, chain: Chain, filename: str):
        """Save a single chain with proper PDB atom name formatting.
        
        Args:
            chain: Chain object to save
            filename: Output PDB filename
        """
        with open(filename, 'w') as f:
            acount = 1
            a2i = {}
            for r, a in chain.atoms.items():
                if a.occ > 0:
                    element = self.i2a[a.element] if a.element in self.i2a.keys() else 'X'
                    
                    # Format atom name properly - use the full atom name from the 4th element of the atom key
                    atom_name = r[3]  # r is (chain_id, res_num, res_name, atom_name)
                    
                    # Format atom name to 4 characters, left-aligned with proper spacing
                    if len(atom_name) < 4:
                        formatted_atom_name = f" {atom_name:<3s}"
                    else:
                        formatted_atom_name = atom_name[:4]
                    
                    f.write("%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % (
                        "HETATM" if a.hetero == True else "ATOM",
                        acount, 
                        formatted_atom_name,  # Use properly formatted atom name
                        r[2],  # residue name
                        r[0],  # chain ID
                        int(r[1]),  # residue number
                        a.xyz[0], a.xyz[1], a.xyz[2], 
                        a.occ, 0.0, 
                        element, 
                        a.charge
                    ))
                    a2i[r] = acount
                    acount += 1
                    
            for bond in chain.bonds:
                if chain.atoms[bond.a].occ == 0.0:
                    continue
                if chain.atoms[bond.b].occ == 0.0:
                    continue
                if chain.atoms[bond.a].hetero == False and chain.atoms[bond.b].hetero == False:
                    continue
                f.write("%-6s%5d%5d\n" % ("CONECT", a2i[bond.a], a2i[bond.b]))


    #@staticmethod
    def save_all(self, chains: Dict[str, Chain], covale: List[Bond], filename: str):
        """Save multiple chains with proper PDB atom name formatting.
        
        Args:
            chains: Dictionary of chain objects to save
            covale: List of covalent bonds between chains
            filename: Output PDB filename
        """
        with open(filename, 'w') as f:
            acount = 1
            a2i = {}
            for chain_id, chain in chains.items():
                for r, a in chain.atoms.items():
                    if a.occ > 0:
                        element = self.i2a[a.element] if a.element in self.i2a.keys() else 'X'
                        
                        # Format atom name properly - use the full atom name from the 4th element of the atom key
                        atom_name = r[3]  # r is (chain_id, res_num, res_name, atom_name)
                        
                        # Format atom name to 4 characters, left-aligned with proper spacing
                        if len(atom_name) < 4:
                            formatted_atom_name = f" {atom_name:<3s}"
                        else:
                            formatted_atom_name = atom_name[:4]
                        
                        f.write("%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % (
                            "HETATM" if a.hetero == True else "ATOM",
                            acount, 
                            formatted_atom_name,  # Use properly formatted atom name
                            r[2],  # residue name  
                            chain_id,  # chain ID
                            int(r[1]),  # residue number
                            a.xyz[0], a.xyz[1], a.xyz[2], 
                            a.occ, 0.0, 
                            element, 
                            a.charge
                        ))
                        a2i[r] = acount
                        acount += 1
                        
                for bond in chain.bonds:
                    a = chain.atoms[bond.a]
                    b = chain.atoms[bond.b]
                    if a.occ == 0.0 or b.occ == 0.0 or (a.hetero == False and b.hetero == False):
                        continue
                    f.write("%-6s%5d%5d\n" % ("CONECT", a2i[bond.a], a2i[bond.b]))
                f.write('TER\n')
            
            for bond in covale:
                a = chains[bond.a[0]].atoms[bond.a]
                b = chains[bond.b[0]].atoms[bond.b]
                if a.occ == 0.0 or b.occ == 0.0:
                    continue
                f.write("%-6s%5d%5d\n" % ("CONECT", a2i[bond.a], a2i[bond.b]))

    def _detect_unresolved_residues(self, chains: Dict) -> set:
        """Detect residues containing unresolved atoms (at origin with bfactor=0.0).
        
        Args:
            chains: Dictionary of chain_id -> Chain objects
            
        Returns:
            Set of residue keys (chain_id, res_num, res_name) to remove
        """
        unresolved_residues = set()
        
        for chain_id, chain in chains.items():
            for atom_key, atom in chain.atoms.items():
                # Check if atom is unresolved: at origin and bfactor = 0.0
                if atom.element == 1:
                    continue # hydrogens are rarely resolved and thats okay.
                if (np.allclose(atom.xyz, [0.0, 0.0, 0.0], atol=1e-6) and 
                    abs(atom.bfac) < 1e-6):
                    res_key = (chain_id, atom_key[1], atom_key[2])  # (chain_id, res_num, res_name)
                    unresolved_residues.add(res_key)
                    logger.debug(f"Detected unresolved atom in residue {res_key}: {atom_key}")
        
        return unresolved_residues

    def _remove_unresolved_residues(self, chains: Dict, unresolved_residues: set) -> Dict:
        """Remove entire residues that contain unresolved atoms.
        
        Args:
            chains: Dictionary of chain_id -> Chain objects
            unresolved_residues: Set of residue keys to remove
            
        Returns:
            Updated chains dictionary with unresolved residues removed
        """
        if not unresolved_residues:
            return chains
        
        updated_chains = {}
        total_removed_atoms = 0
        
        for chain_id, chain in chains.items():
            # Filter atoms and residues
            filtered_atoms = {}
            filtered_residues = {}
            
            for atom_key, atom in chain.atoms.items():
                res_key = (chain_id, atom_key[1], atom_key[2])
                if res_key not in unresolved_residues:
                    filtered_atoms[atom_key] = atom
                else:
                    total_removed_atoms += 1
            
            for res_key, residue in chain.residues.items():
                if residue is None:
                    continue

                chain_res_key = (chain_id, res_key, residue.name)  # (chain_id, res_num, res_name)
                if chain_res_key not in unresolved_residues:
                    filtered_residues[res_key] = residue
                else:
                    pass
            
            # Filter bonds to remove any involving removed atoms
            filtered_bonds = [
                bond for bond in chain.bonds
                if (bond.a in filtered_atoms and bond.b in filtered_atoms)
            ]
            
            # Create updated chain
            updated_chains[chain_id] = chain._replace(
                atoms=filtered_atoms,
                residues=filtered_residues,
                bonds=filtered_bonds
            )
        
        if total_removed_atoms > 0:
            logger.info(f"Removed {len(unresolved_residues)} residues containing {total_removed_atoms} unresolved atoms")
        
        return updated_chains


    def get_metal_sites(self, 
                    parsed_data: Union[Dict, str], 
                    cutoff_distance: float = 6.0,
                    coordination_distance: float = 2.0,
                    merge_threshold: float = 10.0,
                    max_atoms_per_site: int = None,
                    skip_sites_with_entities: Union[List[str], str] = None,
                    min_amino_acids: int = None,
                    min_coordinating_amino_acids: int = None,
                    max_water_bfactor: float = None,
                    backbone_treatment: str = 'bound',
                    edqualitymapper: 'EDQualityMapper' = None
                    ) -> List[Dict]:
        """Extract metal binding sites from protein structure with unresolved atom handling.
        
        Args:
            parsed_data: Either output from parse() or filename to parse
            cutoff_distance: Distance in Angstroms to include residues around metal
            coordination_distance: Distance in Angstroms for coordination shell analysis
            merge_threshold: If metals are closer than this, handle as single site
            max_atoms_per_site: Maximum heavy atoms per binding site (optional)
            skip_sites_with_entities: List of entity names to avoid (skip entire site if present),
                                    or 'non_metal' to only allow amino acids, water, and metals
            min_amino_acids: Minimum number of amino acid residues required in site (optional)
            min_coordinating_amino_acids: Minimum number of amino acid residues within coordination_distance (optional)
            max_water_bfactor: Maximum B-factor for water oxygen atoms. Water molecules with
                            oxygen B-factors above this threshold are excluded (optional)
            backbone_treatment: Treatment of backbone atoms - 'bound' (default), 'free', or 'ca_only'
            
        Returns:
            List of metal site dictionaries
        """
        # Validate backbone_treatment parameter
        if backbone_treatment not in ['bound', 'free', 'ca_only']:
            raise ValueError("backbone_treatment must be one of 'bound', 'free', or 'ca_only'")
        
        # Parse input if filename provided
        if isinstance(parsed_data, str):
            chains, assemblies, covalent, meta = self.parse(parsed_data)
        else:
            chains, assemblies, covalent, meta = parsed_data

        # remove all
        unresolved_residues = self._detect_unresolved_residues(chains)
        original_chains_with_unresolved = copy.deepcopy(chains)  # Keep copy with unresolved
        
        if unresolved_residues:
            chains = self._remove_unresolved_residues(chains, unresolved_residues)

        
        metal_sites = []
        
        # Find all metal atoms across all chains
        all_metal_atoms = []
        for chain_id, chain in chains.items():
            for atom_key, atom in chain.atoms.items():
                if atom.metal and atom.element > 1:
                    all_metal_atoms.append({
                        'chain_id': chain_id,
                        'atom_key': atom_key,
                        'atom': atom,
                        'element': atom.element,
                        'coords': atom.xyz
                    })
        
        if not all_metal_atoms:
            return []
        
        # Group nearby metals using distance clustering
        metal_clusters = self._cluster_metals(all_metal_atoms, merge_threshold)
        
        # Extract binding site for each metal cluster
        for cluster in metal_clusters:
            site_data = self._extract_binding_site(
                cluster, chains, cutoff_distance, coordination_distance, 
                max_atoms_per_site, max_water_bfactor, backbone_treatment
            )
            if site_data:
                unresolved_count = self._count_unresolved_in_site(
                    cluster, original_chains_with_unresolved, unresolved_residues, cutoff_distance
                )
                site_data['n_unresolved_removed'] = unresolved_count
                
                # Apply post-processing filters
                if self._should_include_site(site_data, skip_sites_with_entities, min_amino_acids, min_coordinating_amino_acids):
                    metal_sites.append(site_data)

                    # Update site data with original file metadata
                    site_data['id'] = meta['id'] if meta else 'unknown'
                    site_data['resolution'] = meta.get('resolution', None) if meta else None
                    site_data['max_rczd'] = None

                    # check metal site for quality
                    if edqualitymapper is not None:
                        site_data['max_rczd'] = self._get_max_rczd(
                            site_data, edqualitymapper
                        )

        return metal_sites
    
    def _should_include_site(self, 
                            site_data: Dict, 
                            skip_entities: Union[List[str], str] = None,
                            min_amino_acids: int = None,
                            min_coordinating_amino_acids: int = None) -> bool:
        """Determine if a metal binding site should be included based on filtering criteria.
        
        Args:
            site_data: Metal binding site data dictionary
            skip_entities: List of entity names to avoid (skip site if any present), 
                        or 'non_metal' to only allow amino acids, water, and metals
            min_amino_acids: Minimum number of amino acid residues required in site
            min_coordinating_amino_acids: Minimum number of amino acid residues within coordination distance
            
        Returns:
            True if site should be included, False otherwise
        """
        # Check for entities to skip
        if skip_entities:
            for res_key in site_data['nearby_residues']:
                res_name = res_key[2]  # residue name is at index 2
                if isinstance(skip_entities, list):
                    if res_name in skip_entities:
                        logger.debug(f"Skipping site due to entity: {res_name}")
                        return False
                elif isinstance(skip_entities, str):
                    assert skip_entities == 'non_metal'
                    # Only allow amino acids, water, and metals. No other ligands
                    if res_name not in RESNAME_3LETTER and res_name not in ['HOH', 'WAT', 'H2O']:
                        # Check if this residue contains metal atoms
                        residue_atoms = site_data['nearby_residues'][res_key]
                        is_metal_residue = False
                        for atom_key in residue_atoms:
                            atom = site_data['site_chain'].atoms[atom_key]
                            if atom.metal:
                                is_metal_residue = True
                                break
                        
                        if not is_metal_residue:
                            logger.debug(f"Skipping site due to non-metal entity: {res_name}")
                            return False
        
        # Check minimum total amino acid requirement
        if min_amino_acids is not None:
            # Count amino acid residues using RESNAME_3LETTER
            amino_acid_count = 0
            for res_key in site_data['nearby_residues']:
                res_name = res_key[2]  # residue name is at index 2
                if res_name in RESNAME_3LETTER:
                    amino_acid_count += 1
            
            if amino_acid_count < min_amino_acids:
                logger.debug(f"Skipping site due to insufficient amino acids: {amino_acid_count} < {min_amino_acids}")
                return False
        
        # Check minimum coordinating amino acid requirement
        if min_coordinating_amino_acids is not None:
            # Count coordinating amino acid residues
            coordinating_amino_acid_count = 0
            for res_key in site_data['coordinating_residues']:
                res_name = res_key[2]  # residue name is at index 2
                if res_name in RESNAME_3LETTER:
                    coordinating_amino_acid_count += 1
            
            if coordinating_amino_acid_count < min_coordinating_amino_acids:
                logger.debug(f"Skipping site due to insufficient coordinating amino acids: {coordinating_amino_acid_count} < {min_coordinating_amino_acids}")
                return False
        
        return True

    def _cluster_metals(self, metal_atoms: List[Dict], threshold: float) -> List[List[Dict]]:
        """Cluster metal atoms that are within threshold distance.
        
        Args:
            metal_atoms: List of metal atom dictionaries
            threshold: Distance threshold for clustering
            
        Returns:
            List of metal clusters (each cluster is a list of metal atoms)
        """
        if not metal_atoms:
            return []
        
        # Simple distance-based clustering
        clusters = []
        unassigned = metal_atoms.copy()
        
        while unassigned:
            # Start new cluster with first unassigned metal
            current_cluster = [unassigned.pop(0)]
            
            # Find all metals within threshold of any metal in current cluster
            changed = True
            while changed:
                changed = False
                to_remove = []
                
                for i, metal in enumerate(unassigned):
                    for cluster_metal in current_cluster:
                        dist = np.linalg.norm(
                            np.array(metal['coords']) - np.array(cluster_metal['coords'])
                        )
                        if dist <= threshold:
                            current_cluster.append(metal)
                            to_remove.append(i)
                            changed = True
                            break
                
                # Remove assigned metals (reverse order to maintain indices)
                for i in reversed(to_remove):
                    unassigned.pop(i)
            
            clusters.append(current_cluster)
        
        return clusters

    def _extract_binding_site(self, 
                            metal_cluster: List[Dict], 
                            chains: Dict, 
                            cutoff: float,
                            coordination_distance: float,
                            max_atoms: int = None,
                            max_water_bfactor: float = None,
                            backbone_treatment: str = 'bound') -> Dict:
        """Extract binding site around a metal cluster including complete residues.
        Explicitly excludes hydrogen atoms from the binding site.
        
        Args:
            metal_cluster: List of metal atoms in cluster
            chains: All chains from parsed structure
            cutoff: Distance cutoff for including residues
            coordination_distance: Distance cutoff for coordination analysis
            max_atoms: Maximum number of heavy atoms allowed in binding site (optional)
            max_water_bfactor: Maximum B-factor for water oxygen atoms (optional)
            backbone_treatment: Treatment of backbone atoms - 'bound', 'free', or 'ca_only'
            
        Returns:
            Dictionary with metal site information or None if site should be excluded
        """
        # Calculate center of metal cluster
        metal_coords = np.array([m['coords'] for m in metal_cluster])
        center = np.mean(metal_coords, axis=0)
        
        # First pass: identify residues with any heavy atoms within cutoff
        nearby_residues_set = set()
        
        for chain_id, chain in chains.items():
            for atom_key, atom in chain.atoms.items():
                # Skip hydrogen atoms explicitly (element 1 is hydrogen)
                if atom.element == 1:
                    continue
                    
                atom_coords = np.array(atom.xyz)
                
                # Check distance to any metal in cluster
                min_dist = min(
                    np.linalg.norm(atom_coords - np.array(m['coords']))
                    for m in metal_cluster
                )
                
                if min_dist <= cutoff:
                    res_key = (atom_key[0], atom_key[1], atom_key[2])  # chain, res_num, res_name
                    nearby_residues_set.add(res_key)
        
        # Second pass: collect all heavy atoms from identified residues
        nearby_residues = {}  # res_key -> list of atom_keys
        nearby_atoms = {}     # atom_key -> atom
        
        for chain_id, chain in chains.items():
            for atom_key, atom in chain.atoms.items():
                # Skip hydrogen atoms
                if atom.element == 1:
                    continue
                
                res_key = (atom_key[0], atom_key[1], atom_key[2])
                if res_key in nearby_residues_set:
                    # Apply water B-factor filtering if specified
                    if self._should_exclude_water_residue(res_key, chain, max_water_bfactor):
                        continue
                    
                    if res_key not in nearby_residues:
                        nearby_residues[res_key] = []
                    nearby_residues[res_key].append(atom_key)
                    nearby_atoms[atom_key] = atom
        
        # Remove any residues that ended up with no atoms after filtering
        nearby_residues = {k: v for k, v in nearby_residues.items() if v}
        
        if not nearby_residues:
            return None
        
        # NEW: Calculate coordinating residues (within coordination_distance)
        coordinating_residues = set()
        
        for res_key, atom_list in nearby_residues.items():
            res_name = res_key[2]
            if res_name in RESNAME_3LETTER:  # Only analyze amino acids for coordination
                for atom_key in atom_list:
                    atom = nearby_atoms[atom_key]
                    if atom.element > 1:  # Heavy atoms only
                        atom_coords = np.array(atom.xyz)
                        min_dist_to_metal = min(
                            np.linalg.norm(atom_coords - np.array(m['coords']))
                            for m in metal_cluster
                        )
                        if min_dist_to_metal <= coordination_distance:
                            coordinating_residues.add(res_key)
                            break  # Found at least one coordinating atom in this residue
        
        # Apply atom limit if specified
        if max_atoms is not None and len(nearby_atoms) > max_atoms:
            nearby_atoms, nearby_residues = self._apply_atom_limit(
                nearby_residues, nearby_atoms, max_atoms, metal_cluster, chains
            )
            
            # Recalculate coordinating residues after atom limit is applied
            coordinating_residues = set()
            for res_key, atom_list in nearby_residues.items():
                res_name = res_key[2]
                if res_name in RESNAME_3LETTER:
                    for atom_key in atom_list:
                        if atom_key in nearby_atoms:  # Check if atom survived filtering
                            atom = nearby_atoms[atom_key]
                            if atom.element > 1:
                                atom_coords = np.array(atom.xyz)
                                min_dist_to_metal = min(
                                    np.linalg.norm(atom_coords - np.array(m['coords']))
                                    for m in metal_cluster
                                )
                                if min_dist_to_metal <= coordination_distance:
                                    coordinating_residues.add(res_key)
                                    break
        
        # Create new chain with only the binding site
        site_chain, old_to_new_resid = self._create_site_chain(
            metal_cluster, nearby_residues, nearby_atoms, chains, backbone_treatment
        )

        # remove bonds, and other attributes of metals that are clearly wrong.
        site_chain = self.clean_metal_bonding_patterns(site_chain)

        coordinating_residues_new = []
        for res_key in coordinating_residues:
            if res_key in old_to_new_resid:
                new_res_key = old_to_new_resid[res_key]
                new_res_key = (site_chain.id, new_res_key, res_key[2])
                coordinating_residues_new.append(new_res_key)
            else:
                raise ValueError(
                    f"Coordinating residue {res_key} not found in new site chain"
                )
        
        return {
            'metal_atoms': metal_cluster,
            'center': center,
            'nearby_residues': nearby_residues,
            'coordinating_residues': coordinating_residues_new,
            'site_chain': site_chain,
            'n_metals': len(metal_cluster),
            'n_residues': len(nearby_residues),
            'n_coordinating_residues': len(coordinating_residues),
            'n_atoms': len(nearby_atoms),
            'atom_limit_applied': max_atoms is not None and len(nearby_atoms) > max_atoms,
            'complete_residues': True,
            'heavy_atoms_only': True,
            'water_bfactor_filtered': max_water_bfactor is not None,
            'backbone_treatment': backbone_treatment,
            'coordination_distance': coordination_distance,
            'n_unresolved_removed': 0  # Placeholder, filled by caller
        }

    def _count_unresolved_in_site(self,
                                metal_cluster: List[Dict],
                                original_chains: Dict,
                                unresolved_residues: set,
                                cutoff_distance: float) -> int:
        """Count how many unresolved residues would have been in the metal site.
        
        Args:
            metal_cluster: List of metal atoms in cluster
            original_chains: Original chains before unresolved removal
            unresolved_residues: Set of unresolved residue keys
            cutoff_distance: Distance cutoff for site inclusion
            
        Returns:
            Number of unresolved residues within cutoff distance
        """
        unresolved_in_site = set()
        
        # For each unresolved residue, check if any atom would be in site
        for res_key in unresolved_residues:
            chain_id, res_num, res_name = res_key
            
            # Find atoms from this residue in original chains
            if chain_id in original_chains:
                chain = original_chains[chain_id]
                for atom_key, atom in chain.atoms.items():
                    # Check if this atom belongs to the unresolved residue
                    if (atom_key[0] == chain_id and 
                        atom_key[1] == res_num and 
                        atom_key[2] == res_name and
                        atom.element > 1):  # Heavy atoms only
                        
                        atom_coords = np.array(atom.xyz)

                        # don't consider this one if its the unrewsolved atom
                        if np.allclose(atom_coords, [0.0, 0.0, 0.0], atol=1e-6) and abs(atom.bfac) < 1e-6:
                            continue
                        
                        # Check distance to any metal in cluster
                        min_dist = min(
                            np.linalg.norm(atom_coords - np.array(m['coords']))
                            for m in metal_cluster
                        )
                        
                        if min_dist <= cutoff_distance:
                            unresolved_in_site.add(res_key)
                            break  # Found at least one atom in range, count residue
        
        return len(unresolved_in_site)
    
    def _apply_atom_limit(self,
                        nearby_residues: Dict,
                        nearby_atoms: Dict,
                        max_atoms: int,
                        metal_cluster: List[Dict],
                        chains: Dict) -> Tuple[Dict, Dict]:
        """Apply atom limit by removing complete residues farthest from metals.
        Only considers heavy atoms in the atom counting.
        
        Args:
            nearby_residues: Dictionary of residue keys to atom lists
            nearby_atoms: Dictionary of all heavy atoms
            max_atoms: Maximum heavy atoms to keep
            metal_cluster: Metal atoms (always preserved)
            chains: Original chain objects
            
        Returns:
            Tuple of (filtered_nearby_atoms, filtered_nearby_residues)
        """
        # Separate metal residues from other residues
        metal_atom_keys = {tuple(m['atom_key']) for m in metal_cluster}
        metal_residues = set()
        non_metal_residues = []
        
        for res_key, atom_list in nearby_residues.items():
            is_metal_residue = any(tuple(ak) in metal_atom_keys for ak in atom_list)
            if is_metal_residue:
                metal_residues.add(res_key)
            else:
                non_metal_residues.append(res_key)
        
        # Count heavy atoms in metal residues (always keep these)
        metal_atom_count = sum(
            len([ak for ak in atom_list if nearby_atoms[ak].element > 1])  # Count only heavy atoms
            for res_key, atom_list in nearby_residues.items()
            if res_key in metal_residues
        )
        
        remaining_budget = max_atoms - metal_atom_count
        if remaining_budget <= 0:
            # Only keep metal residues
            filtered_residues = {k: v for k, v in nearby_residues.items() if k in metal_residues}
            filtered_atoms = {
                ak: nearby_atoms[ak] 
                for atom_list in filtered_residues.values() 
                for ak in atom_list
                if nearby_atoms[ak].element > 1  # Only heavy atoms
            }
            return filtered_atoms, filtered_residues
        
        # Calculate average distance per non-metal residue to metals (heavy atoms only)
        residue_distances = []
        for res_key in non_metal_residues:
            atom_list = nearby_residues[res_key]
            distances = []
            heavy_atom_count = 0
            
            for atom_key in atom_list:
                atom = nearby_atoms[atom_key]
                if atom.element > 1:  # Only consider heavy atoms
                    atom_coords = np.array(atom.xyz)
                    min_dist = min(
                        np.linalg.norm(atom_coords - np.array(m['coords']))
                        for m in metal_cluster
                    )
                    distances.append(min_dist)
                    heavy_atom_count += 1
            
            if heavy_atom_count > 0:  # Only include residues with heavy atoms
                avg_dist = np.mean(distances)
                residue_distances.append((res_key, avg_dist, heavy_atom_count))
        
        # Sort by distance (closest first)
        residue_distances.sort(key=lambda x: x[1])
        
        # Add complete residues until budget exhausted
        selected_residues = {k: v for k, v in nearby_residues.items() if k in metal_residues}
        current_atom_count = metal_atom_count
        
        for res_key, avg_dist, heavy_atom_count in residue_distances:
            if current_atom_count + heavy_atom_count <= max_atoms:
                selected_residues[res_key] = nearby_residues[res_key]
                current_atom_count += heavy_atom_count
            else:
                break  # Can't fit this complete residue
        
        # Build final atom dictionary (heavy atoms only)
        filtered_atoms = {
            ak: nearby_atoms[ak] 
            for atom_list in selected_residues.values() 
            for ak in atom_list
            if nearby_atoms[ak].element > 1  # Only heavy atoms
        }
        logger.debug(f"Applied atom limit: removed {len(nearby_residues) - len(selected_residues)} residues, orignal n atoms = {len(nearby_atoms)}, filtered n atoms = {len(filtered_atoms)}")
                         
        return filtered_atoms, selected_residues

    def _create_site_chain(self, 
                        metal_cluster: List[Dict],
                        nearby_residues: Dict,
                        nearby_atoms: Dict,
                        original_chains: Dict,
                        backbone_treatment: str = 'bound') -> 'Chain':
        """Create a new Chain object containing only the metal binding site.
        Only includes heavy atoms in the site chain.
        
        Args:
            metal_cluster: Metal atoms in site
            nearby_residues: Residues within cutoff
            nearby_atoms: All heavy atoms within cutoff
            original_chains: Original chain objects
            backbone_treatment: Treatment of backbone atoms - 'bound', 'free', or 'ca_only'
            
        Returns:
            New Chain object with metal site (heavy atoms only)
        """
        
        # Create mapping from old atom keys to new atom keys
        old_to_new_atom_key = {}
        new_to_old_atom_key = {}
        old_to_new_resid = {}
        
        # Create site-specific residue objects with renumbering
        site_residues = {}
        site_bonds = []
        new_res_num = 1  # Start residue numbering from 1
        
        for res_key in nearby_residues:
            chain_id = res_key[0]
            original_chain = original_chains[chain_id]
            old_res_pos = res_key[1]
            res_name = res_key[2]
            
            if old_res_pos in original_chain.residues:
                # Copy residue but filter atoms to heavy atoms only
                original_res = original_chain.residues[old_res_pos]
                site_atoms = [ak for ak in nearby_residues[res_key] 
                            if ak in nearby_atoms and nearby_atoms[ak].element > 1]
                
                # Create new atom keys with renumbered residue and chain ID "A"
                new_residue_atoms = {}
                for old_atom_key in site_atoms:
                    if old_atom_key in nearby_atoms:
                        atom_name = old_atom_key[3]  # atom name is the 4th element
                        new_atom_key = ("A", str(new_res_num), res_name, atom_name)
                        
                        # Store mappings
                        old_to_new_atom_key[old_atom_key] = new_atom_key
                        new_to_old_atom_key[new_atom_key] = old_atom_key
                        
                        # Copy atom with new key
                        atom = nearby_atoms[old_atom_key]
                        new_residue_atoms[atom_name] = atom._replace(name=new_atom_key)
                
                # Filter bonds to only those between heavy atoms in the site, with new keys
                filtered_bonds = []
                for bond in original_res.bonds:
                    old_a_key = (chain_id, old_res_pos, res_name, bond.a)
                    old_b_key = (chain_id, old_res_pos, res_name, bond.b)
                    
                    if (old_a_key in old_to_new_atom_key and old_b_key in old_to_new_atom_key):
                        new_a_key = old_to_new_atom_key[old_a_key]
                        new_b_key = old_to_new_atom_key[old_b_key]
                        
                        # Create bond with new atom keys
                        new_bond = bond._replace(
                            a=new_a_key,
                            b=new_b_key
                        )
                        filtered_bonds.append(new_bond)
                
                site_residues[str(new_res_num)] = Residue(
                    name=original_res.name,
                    atoms=new_residue_atoms,
                    bonds=filtered_bonds,
                    automorphisms=original_res.automorphisms,  # Will be updated below
                    chirals=original_res.chirals,              # Will be updated below
                    planars=original_res.planars,              # Will be updated below
                    alternatives=original_res.alternatives
                )

                old_to_new_resid[res_key] = str(new_res_num)
                new_res_num += 1
                
        
        # Filter inter-residue bonds to site (heavy atoms only) and update keys
        for chain in original_chains.values():
            for bond in chain.bonds:
                if (bond.a in old_to_new_atom_key and bond.b in old_to_new_atom_key):
                    new_a_key = old_to_new_atom_key[bond.a]
                    new_b_key = old_to_new_atom_key[bond.b]
                    
                    new_bond = bond._replace(
                        a=new_a_key,
                        b=new_b_key
                    )
                    site_bonds.append(new_bond)
        
        # Create new atoms dict with updated keys
        new_atoms = {}
        for old_atom_key, new_atom_key in old_to_new_atom_key.items():
            if old_atom_key in nearby_atoms:
                atom = nearby_atoms[old_atom_key]
                new_atoms[new_atom_key] = atom._replace(name=new_atom_key)
        
        # Apply backbone treatment filtering
        filtered_atoms, filtered_bonds = self._filter_backbone(
            new_atoms, site_bonds, backbone_treatment
        )
        
        # === PRESERVE CHIRALS, PLANARS, AND AUTOMORPHISMS WITH KEY REMAPPING ===
        
        def remap_constraint_keys(constraint_list, old_to_new_mapping, filtered_atoms_set):
            """Remap atom keys in constraints and filter to only include valid ones."""
            remapped_constraints = []
            
            for constraint in constraint_list:
                # Try to remap all atom keys in the constraint
                remapped_constraint = []
                all_atoms_present = True
                
                for old_atom_key in constraint:
                    if old_atom_key in old_to_new_mapping:
                        new_atom_key = old_to_new_mapping[old_atom_key]
                        if new_atom_key in filtered_atoms_set:
                            remapped_constraint.append(new_atom_key)
                        else:
                            all_atoms_present = False
                            break
                    else:
                        all_atoms_present = False
                        break
                
                # Only include constraint if all atoms are present in filtered site
                if all_atoms_present and len(remapped_constraint) == len(constraint):
                    remapped_constraints.append(remapped_constraint)
            
            return remapped_constraints
        
        # Collect and remap chirals from original chains
        site_chirals = []
        for chain in original_chains.values():
            remapped_chirals = remap_constraint_keys(
                chain.chirals, old_to_new_atom_key, set(filtered_atoms.keys())
            )
            site_chirals.extend(remapped_chirals)
        
        # Collect and remap planars from original chains  
        site_planars = []
        for chain in original_chains.values():
            remapped_planars = remap_constraint_keys(
                chain.planars, old_to_new_atom_key, set(filtered_atoms.keys())
            )
            site_planars.extend(remapped_planars)
        
        # Collect and remap automorphisms from original chains
        site_automorphisms = []
        for chain in original_chains.values():
            for auto_group in chain.automorphisms:
                remapped_auto_group = []
                for auto_set in auto_group:
                    remapped_auto_set = []
                    all_atoms_present = True
                    
                    for old_atom_key in auto_set:
                        if old_atom_key in old_to_new_atom_key:
                            new_atom_key = old_to_new_atom_key[old_atom_key]
                            if new_atom_key in filtered_atoms:
                                remapped_auto_set.append(new_atom_key)
                            else:
                                all_atoms_present = False
                                break
                        else:
                            all_atoms_present = False
                            break
                    
                    if all_atoms_present and len(remapped_auto_set) == len(auto_set):
                        remapped_auto_group.append(remapped_auto_set)
                
                if remapped_auto_group:
                    site_automorphisms.append(remapped_auto_group)
        
        # Update residue-level constraints with new keys
        for res_num, residue in site_residues.items():
            # Filter automorphisms to only those atoms in this residue
            res_automorphisms = []
            for auto_group in residue.automorphisms:
                filtered_auto_group = []
                for auto_set in auto_group:
                    # Check if this automorphism set contains atoms from this residue
                    filtered_auto_set = [
                        atom_name for atom_name in auto_set 
                        if atom_name in residue.atoms
                    ]
                    if len(filtered_auto_set) == len(auto_set):
                        filtered_auto_group.append(filtered_auto_set)
                
                if filtered_auto_group:
                    res_automorphisms.append(filtered_auto_group)
            
            # Similar filtering for chirals and planars at residue level
            res_chirals = [
                chiral for chiral in residue.chirals
                if all(atom_name in residue.atoms for atom_name in chiral)
            ]
            
            res_planars = [
                planar for planar in residue.planars  
                if all(atom_name in residue.atoms for atom_name in planar)
            ]
            
            # Update the residue with filtered constraints
            site_residues[res_num] = residue._replace(
                automorphisms=res_automorphisms,
                chirals=res_chirals,
                planars=res_planars
            )
        
        # === END PRESERVE CONSTRAINTS ===
        
        # Create site chain with properly remapped keys and preserved constraints
        site_chain = Chain(
            id=f"A",
            type="metal_binding_site",
            sequence="",
            residues=site_residues,
            atoms=filtered_atoms,      # Properly renumbered and filtered
            bonds=filtered_bonds,      # Properly renumbered and filtered  
            chirals=site_chirals,      # Preserved and remapped from original chains
            planars=site_planars,      # Preserved and remapped from original chains
            automorphisms=site_automorphisms  # Preserved and remapped from original chains
        )
        
        return site_chain, old_to_new_resid
    
    def _should_exclude_water_residue(self, 
                                    res_key: Tuple[str, str, str], 
                                    chain: 'Chain', 
                                    max_water_bfactor: float = None) -> bool:
        """Determine if a water residue should be excluded based on B-factor filtering.
        
        Args:
            res_key: Tuple of (chain_id, res_num, res_name)
            chain: Chain object containing the residue
            max_water_bfactor: Maximum B-factor threshold for water oxygen atoms
            
        Returns:
            True if water residue should be excluded, False otherwise
        """
        # If no B-factor filtering, don't exclude
        if max_water_bfactor is None:
            return False
        
        chain_id, res_num, res_name = res_key
        
        # Check if this is a water molecule
        if res_name not in ['HOH', 'WAT', 'H2O']:
            return False
        
        # Find oxygen atom(s) in this water residue and check B-factor
        oxygen_bfactors = []
        for atom_key, atom in chain.atoms.items():
            if (atom_key[0] == chain_id and 
                atom_key[1] == res_num and 
                atom_key[2] == res_name and
                atom.element == 8):  # Oxygen has atomic number 8
                
                oxygen_bfactors.append(atom.bfac)
        
        # If we found oxygen atoms, check if any exceed the threshold
        if oxygen_bfactors:
            max_oxygen_bfactor = max(oxygen_bfactors)
            return max_oxygen_bfactor > max_water_bfactor
        
        # If no oxygen found (shouldn't happen for water), don't exclude
        return False
    
    def _filter_backbone(self, 
                    atoms_dict: Dict, 
                    bonds_list: List, 
                    backbone_treatment: str) -> Tuple[Dict, List]:
        """Filter backbone atoms and bonds based on treatment option.
        Only applies to protein peptides, nucleotides are left unchanged.
        
        Args:
            atoms_dict: Dictionary of atom_key -> Atom
            bonds_list: List of Bond objects
            backbone_treatment: 'bound', 'free', or 'ca_only'
            
        Returns:
            Tuple of (filtered_atoms_dict, filtered_bonds_list)
        """
        if backbone_treatment == 'bound':
            # No filtering, return as is
            return atoms_dict, bonds_list
        
        # Define backbone atom names for amino acids only
        protein_backbone_atoms = {'N', 'CA', 'C', 'O', 'OXT'}
        
        filtered_atoms = {}
        
        # Filter atoms based on backbone treatment
        for atom_key, atom in atoms_dict.items():
            chain_id, res_num, res_name, atom_name = atom_key
            
            # Check if this is a protein backbone atom
            is_protein_backbone = (res_name in RESNAME_3LETTER and atom_name in protein_backbone_atoms)
            
            if backbone_treatment == 'ca_only':
                # Only keep CA atoms from protein backbone
                if is_protein_backbone and atom_name != 'CA':
                    continue  # Skip non-CA backbone atoms
            
            elif backbone_treatment == 'free':
                # Keep all backbone atoms (they'll be disconnected by bond filtering)
                pass
            
            # Keep the atom
            filtered_atoms[atom_key] = atom
        
        # Filter bonds based on backbone treatment
        filtered_bonds = []
        
        for bond in bonds_list:
            # Check if both atoms still exist after atom filtering
            if bond.a not in filtered_atoms or bond.b not in filtered_atoms:
                continue
            
            # Get atom information
            atom_a_key = bond.a
            atom_b_key = bond.b
            atom_a_name = atom_a_key[3]
            atom_b_name = atom_b_key[3]
            res_a = (atom_a_key[0], atom_a_key[1], atom_a_key[2])  # chain, res_num, res_name
            res_b = (atom_b_key[0], atom_b_key[1], atom_b_key[2])
            res_name_a = atom_a_key[2]
            res_name_b = atom_b_key[2]
            
            # Check if this is a peptide bond (inter-residue backbone connection between amino acids)
            is_peptide_bond = (
                not bond.intra and  # Inter-residue bond
                res_a != res_b and  # Different residues
                res_name_a in RESNAME_3LETTER and res_name_b in RESNAME_3LETTER and  # Both are amino acids
                ((atom_a_name == 'C' and atom_b_name == 'N') or 
                (atom_a_name == 'N' and atom_b_name == 'C'))
            )
            
            if backbone_treatment == 'free' and is_peptide_bond:
                # Skip peptide bonds to disconnect backbone
                continue
            
            # Keep the bond
            filtered_bonds.append(bond)
        
        return filtered_atoms, filtered_bonds
    
    def _get_max_rczd(self, site_data: Dict, edqualitymapper: 'EDQualityMapping') -> float:
        """
        Get maximum RSZD (electron density quality) value for metals in the site.
        
        Args:
            site_data: Dictionary containing metal site information with 'metal_atoms' and 'id'
            edqualitymapper: EDQualityMapping instance for quality assessment
            
        Returns:
            Optional[float]: Maximum RSZD value observed for metals in site, None if no quality data found
        """
        # Get PDB ID from site metadata
        pdb_id = site_data.get('id')
        if not pdb_id:
            logger.warning("No PDB ID available for metal quality assessment")
            return None
        
        # Extract metal atoms from site data
        metal_atoms = site_data.get('metal_atoms', [])
        if not metal_atoms:
            logger.debug(f"No metal atoms found in site for PDB {pdb_id}")
            return None
        
        # Collect RSZD values for all metals in site
        rszd_values = []
        
        for metal_atom_data in metal_atoms:
            # Extract metal information
            atom = metal_atom_data.get('atom')
            if not atom:
                continue
                
            # Get element symbol from atomic number
            element_symbol = I2E.get(atom.element, 'UNK')
            
            if element_symbol == 'UNK':
                continue
            
            # Get coordinates
            coordinates = atom.xyz
            coord_tuple = (float(coordinates[0]), float(coordinates[1]), float(coordinates[2]))
            
            # Query the quality mapping
            quality_entry = edqualitymapper.get_metal_quality(
                pdb_id=pdb_id,
                metal_symbol=element_symbol,
                coordinates=coord_tuple
            )
            
            if quality_entry is not None:
                rszd_values.append(quality_entry.rszd)
        
        # Return maximum RSZD value if any found, otherwise None
        return max(rszd_values) if rszd_values else None


    def clean_metal_bonding_patterns(self, chain: 'Chain') -> 'Chain':
        """
        Clean metal bonding patterns by removing bonds, hybridization, and hydrogen counts
        for all atoms in residues containing metals.
        
        This method:
        1. Identifies residues containing metal atoms
        2. Sets hybridization to None and nhyd to 0 for all atoms in metal-containing residues
        3. Removes all bonds involving atoms in metal-containing residues
        4. Removes chirals and planars containing atoms from metal-containing residues
        
        Args:
            chain: Chain object to clean
            
        Returns:
            New Chain object with cleaned metal bonding patterns
        """
        import copy
        
        # Create a copy of the chain to avoid modifying the original
        cleaned_chain = copy.deepcopy(chain)
        
        # Step 1: Identify residues containing metal atoms
        metal_residue_keys = set()
        metal_containing_atom_keys = set()
        
        for atom_key, atom in cleaned_chain.atoms.items():
            if atom.metal:
                # Get residue key: (chain_id, res_num, res_name)
                res_key = (atom_key[0], atom_key[1], atom_key[2])
                metal_residue_keys.add(res_key)
        
        # Collect all atom keys from metal-containing residues
        for atom_key, atom in cleaned_chain.atoms.items():
            res_key = (atom_key[0], atom_key[1], atom_key[2])
            if res_key in metal_residue_keys:
                metal_containing_atom_keys.add(atom_key)
        
        # Step 2: Set hybridization to None and nhyd to 0 for all atoms in metal-containing residues
        for atom_key in metal_containing_atom_keys:
            atom = cleaned_chain.atoms[atom_key]
            cleaned_chain.atoms[atom_key] = atom._replace(hyb=None, nhyd=None)
        
        # Step 3: Remove all bonds involving atoms from metal-containing residues
        cleaned_bonds = []
        for bond in cleaned_chain.bonds:
            if bond.a not in metal_containing_atom_keys and bond.b not in metal_containing_atom_keys:
                cleaned_bonds.append(bond)
        
        # Step 4: Remove chirals containing atoms from metal-containing residues
        cleaned_chirals = []
        for chiral in cleaned_chain.chirals:
            has_metal_residue_atom = any(atom_key in metal_containing_atom_keys for atom_key in chiral)
            if not has_metal_residue_atom:
                cleaned_chirals.append(chiral)
        
        # Step 5: Remove planars containing atoms from metal-containing residues
        cleaned_planars = []
        for planar in cleaned_chain.planars:
            has_metal_residue_atom = any(atom_key in metal_containing_atom_keys for atom_key in planar)
            if not has_metal_residue_atom:
                cleaned_planars.append(planar)
        
        # Step 6: Clean automorphisms - remove groups containing atoms from metal-containing residues
        cleaned_automorphisms = []
        for auto_group in cleaned_chain.automorphisms:
            cleaned_auto_group = []
            for auto_set in auto_group:
                has_metal_residue_atom = any(atom_key in metal_containing_atom_keys for atom_key in auto_set)
                if not has_metal_residue_atom:
                    cleaned_auto_group.append(auto_set)
            
            if cleaned_auto_group:
                cleaned_automorphisms.append(cleaned_auto_group)
        
        # Step 7: Update residue-level bonding patterns for metal-containing residues
        cleaned_residues = {}
        for res_pos, residue in cleaned_chain.residues.items():
            if residue is not None:
                res_key = (cleaned_chain.id, res_pos, residue.name)
                
                if res_key in metal_residue_keys:
                    # For metal-containing residues, remove all bonds, chirals, planars
                    cleaned_residues[res_pos] = residue._replace(
                        bonds=[],  # Remove all intra-residue bonds
                        chirals=[],  # Remove all chirals
                        planars=[],  # Remove all planars
                        automorphisms=[]  # Remove all automorphisms
                    )
                else:
                    # For non-metal residues, keep original bonding patterns
                    cleaned_residues[res_pos] = residue
            else:
                cleaned_residues[res_pos] = residue
        
        # Create the cleaned chain
        cleaned_chain = Chain(
            id=cleaned_chain.id,
            type=cleaned_chain.type,
            sequence=cleaned_chain.sequence,
            residues=cleaned_residues,
            atoms=cleaned_chain.atoms,  # Already updated in step 2
            bonds=cleaned_bonds,
            chirals=cleaned_chirals,
            planars=cleaned_planars,
            automorphisms=cleaned_automorphisms
        )
        
        return cleaned_chain

    def get_standard_amino_acids(self) -> Dict[str, Residue]:
        """
        Get a dictionary of all standard amino acid residues with proper geometry.
        Uses the existing molecular library to create Residue objects with zero coordinates.
        
        Returns:
            Dict[str, Residue]: Dictionary mapping 3-letter codes to Residue objects
        """
        from metalsitenn.constants import RESNAME_3LETTER
        import copy
        
        standard_residues = {}
        
        for aa_code in RESNAME_3LETTER:
            res_data = self.getRes(aa_code)
            if res_data is not None and 'res' in res_data:
                # Get the residue and make a deep copy
                residue = copy.deepcopy(res_data['res'])
                
                # Set all coordinates to zero as requested
                for atom_name, atom in residue.atoms.items():
                    residue.atoms[atom_name] = atom._replace(xyz=[0.0, 0.0, 0.0])
                
                standard_residues[aa_code] = residue
        
        return standard_residues


# module level
def get_standard_amino_acids() -> Dict[str, Residue]:
    """
    Get a dictionary of all standard amino acid residues with proper geometry.
    Uses the existing molecular library to create Residue objects with zero coordinates.
    
    Returns:
        Dict[str, Residue]: Dictionary mapping 3-letter codes to Residue objects
    """
    parser = CIFParser()
    standard_residues = parser.get_standard_amino_acids()
    return standard_residues

RESIDUES_TEMPLATES = get_standard_amino_acids()

def mutate_chain(chain: Chain, target_res_num: str, target_res_name: str, new_res_name: str) -> Chain:
    """
    Mutate a residue in a chain to a new amino acid residue type.
    
    Replaces the target residue with a new amino acid from RESIDUES_TEMPLATES,
    positioning all new atoms at the original CA position.
    
    Args:
        chain: Chain object to mutate
        target_res_num: Residue number (as string) to mutate
        target_res_name: Current residue name (3-letter code) to verify target
        new_res_name: New amino acid residue name (3-letter code)
        
    Returns:
        New Chain object with the mutation applied
        
    Raises:
        ValueError: If target residue not found, new residue not in templates, 
                   or target CA atom not found
    """
    # Validate inputs
    if new_res_name not in RESIDUES_TEMPLATES:
        raise ValueError(f"New residue type '{new_res_name}' not found in amino acid templates")
    
    # Find target residue and get CA position
    target_ca_pos = None
    target_res_key = None
    
    for res_key in chain.residues:
        if res_key == target_res_num:
            residue = chain.residues[res_key]
            if residue and residue.name == target_res_name:
                target_res_key = res_key
                # Find CA atom in this residue
                for atom_key, atom in chain.atoms.items():
                    if (atom_key[0] == chain.id and 
                        atom_key[1] == target_res_num and 
                        atom_key[2] == target_res_name and 
                        atom_key[3] == 'CA'):
                        target_ca_pos = atom.xyz
                        break
                break
    
    if target_res_key is None:
        raise ValueError(f"Target residue {target_res_num}:{target_res_name} not found in chain")
    
    if target_ca_pos is None:
        raise ValueError(f"CA atom not found for target residue {target_res_num}:{target_res_name}")
    
    # Get new residue template
    new_residue_template = copy.deepcopy(RESIDUES_TEMPLATES[new_res_name])
    
    # Create new chain components
    new_residues = {}
    new_atoms = {}
    new_bonds = []
    new_chirals = []
    new_planars = []
    new_automorphisms = []
    
    # Copy all residues except the target
    for res_key, residue in chain.residues.items():
        if res_key != target_res_key:
            new_residues[res_key] = copy.deepcopy(residue)
    
    # Copy all atoms except those from target residue
    for atom_key, atom in chain.atoms.items():
        if not (atom_key[0] == chain.id and 
                atom_key[1] == target_res_num and 
                atom_key[2] == target_res_name):
            new_atoms[atom_key] = atom
    
    # Add new atoms from template positioned at CA location (heavy atoms only)
    new_residue_atoms = {}
    for atom_name, template_atom in new_residue_template.atoms.items():
        # Skip hydrogen atoms (element 1 is hydrogen)
        if template_atom.element == 1:
            continue
            
        new_atom_key = (chain.id, target_res_num, new_res_name, atom_name)
        new_atom = template_atom._replace(
            name=new_atom_key,
            xyz=target_ca_pos,  # Position all atoms at original CA position
            hetero=False  # Amino acids are not hetero atoms
        )
        new_atoms[new_atom_key] = new_atom
        new_residue_atoms[atom_name] = new_atom
    
    # Create updated residue object with heavy atoms only and updated atom keys
    new_residues[target_res_key] = new_residue_template._replace(
        name=new_res_name,
        atoms=new_residue_atoms
    )
    
    # Update bonds - remove old bonds involving target residue, add new ones
    old_to_new_atom_keys = {}  # For tracking inter-residue bonds
    
    # Map old target residue atom keys to new ones for inter-residue bond updating (heavy atoms only)
    for atom_name, template_atom in new_residue_template.atoms.items():
        # Skip hydrogens
        if template_atom.element == 1:
            continue
            
        old_key = (chain.id, target_res_num, target_res_name, atom_name)
        new_key = (chain.id, target_res_num, new_res_name, atom_name)
        if atom_name in new_residue_atoms:  # Only if atom exists in new residue (heavy atoms)
            old_to_new_atom_keys[old_key] = new_key
    
    # Copy bonds that don't involve the target residue
    for bond in chain.bonds:
        old_res_a = (bond.a[0], bond.a[1], bond.a[2])  # chain, res_num, res_name
        old_res_b = (bond.b[0], bond.b[1], bond.b[2])
        target_res = (chain.id, target_res_num, target_res_name)
        
        if old_res_a == target_res or old_res_b == target_res:
            # This is a bond involving the target residue
            if not bond.intra:
                # Inter-residue bond - try to preserve if atoms exist in new residue
                new_a = old_to_new_atom_keys.get(bond.a, bond.a)
                new_b = old_to_new_atom_keys.get(bond.b, bond.b)
                
                # Only add bond if both atoms exist in the new structure
                if new_a in new_atoms and new_b in new_atoms:
                    new_bond = bond._replace(a=new_a, b=new_b)
                    new_bonds.append(new_bond)
            # Skip intra-residue bonds from old residue - they'll be replaced
        else:
            # Bond doesn't involve target residue, keep as is
            new_bonds.append(bond)
    
    # Add intra-residue bonds from new residue template (heavy atoms only)
    for template_bond in new_residue_template.bonds:
        # Skip bonds involving hydrogen atoms
        atom_a = new_residue_template.atoms.get(template_bond.a)
        atom_b = new_residue_template.atoms.get(template_bond.b)
        
        if (atom_a and atom_a.element == 1) or (atom_b and atom_b.element == 1):
            continue  # Skip bonds to hydrogen
            
        new_a = (chain.id, target_res_num, new_res_name, template_bond.a)
        new_b = (chain.id, target_res_num, new_res_name, template_bond.b)
        
        # Only add bond if both atoms exist in new_atoms (both are heavy atoms)
        if new_a in new_atoms and new_b in new_atoms:
            new_bond = template_bond._replace(a=new_a, b=new_b)
            new_bonds.append(new_bond)
    
    # Update chirals - remove old ones involving target residue, add new ones
    for chiral in chain.chirals:
        # Check if any atom in chiral involves target residue
        involves_target = any(
            atom_key[0] == chain.id and 
            atom_key[1] == target_res_num and 
            atom_key[2] == target_res_name
            for atom_key in chiral
        )
        
        if not involves_target:
            new_chirals.append(chiral)
    
    # Add chirals from new residue template (heavy atoms only)
    for template_chiral in new_residue_template.chirals:
        # Check if all atoms in chiral are heavy atoms and exist in new residue
        new_chiral = []
        skip_chiral = False
        
        for atom_name in template_chiral:
            template_atom = new_residue_template.atoms.get(atom_name)
            if template_atom and template_atom.element == 1:
                skip_chiral = True  # Skip chirals involving hydrogens
                break
            new_atom_key = (chain.id, target_res_num, new_res_name, atom_name)
            if new_atom_key in new_atoms:
                new_chiral.append(new_atom_key)
            else:
                skip_chiral = True
                break
        
        if not skip_chiral and len(new_chiral) == len(template_chiral):
            new_chirals.append(new_chiral)
    
    # Update planars - remove old ones involving target residue, add new ones
    for planar in chain.planars:
        # Check if any atom in planar involves target residue
        involves_target = any(
            atom_key[0] == chain.id and 
            atom_key[1] == target_res_num and 
            atom_key[2] == target_res_name
            for atom_key in planar
        )
        
        if not involves_target:
            new_planars.append(planar)
    
    # Add planars from new residue template (heavy atoms only)
    for template_planar in new_residue_template.planars:
        # Check if all atoms in planar are heavy atoms and exist in new residue
        new_planar = []
        skip_planar = False
        
        for atom_name in template_planar:
            template_atom = new_residue_template.atoms.get(atom_name)
            if template_atom and template_atom.element == 1:
                skip_planar = True  # Skip planars involving hydrogens
                break
            new_atom_key = (chain.id, target_res_num, new_res_name, atom_name)
            if new_atom_key in new_atoms:
                new_planar.append(new_atom_key)
            else:
                skip_planar = True
                break
        
        if not skip_planar and len(new_planar) == len(template_planar):
            new_planars.append(new_planar)
    
    # Update automorphisms - remove groups involving target residue
    for auto_group in chain.automorphisms:
        new_auto_group = []
        for auto_set in auto_group:
            # Check if this set involves target residue
            involves_target = any(
                atom_key[0] == chain.id and 
                atom_key[1] == target_res_num and 
                atom_key[2] == target_res_name
                for atom_key in auto_set
            )
            
            if not involves_target:
                new_auto_group.append(auto_set)
        
        if new_auto_group:  # Only add if not empty
            new_automorphisms.append(new_auto_group)
    
    # Add automorphisms from new residue template (heavy atoms only)
    for template_auto_group in new_residue_template.automorphisms:
        new_auto_group = []
        for template_auto_set in template_auto_group:
            # Check if all atoms in this set are heavy atoms and exist in new residue
            new_auto_set = []
            skip_set = False
            
            for atom_name in template_auto_set:
                template_atom = new_residue_template.atoms.get(atom_name)
                if template_atom and template_atom.element == 1:
                    skip_set = True  # Skip sets involving hydrogens
                    break
                new_atom_key = (chain.id, target_res_num, new_res_name, atom_name)
                if new_atom_key in new_atoms:
                    new_auto_set.append(new_atom_key)
                else:
                    skip_set = True
                    break
            
            if not skip_set and len(new_auto_set) == len(template_auto_set):
                new_auto_group.append(new_auto_set)
        
        if new_auto_group:
            new_automorphisms.append(new_auto_group)
    
    # Create new chain
    new_chain = Chain(
        id=chain.id,
        type=chain.type,
        sequence=chain.sequence,  # Note: sequence string not updated here
        residues=new_residues,
        atoms=new_atoms,
        bonds=new_bonds,
        chirals=new_chirals,
        planars=new_planars,
        automorphisms=new_automorphisms
    )
    
    return new_chain