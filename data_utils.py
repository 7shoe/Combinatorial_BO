import os
import pandas as pd
import numpy as np
import math
from pathlib import Path
from matplotlib import pyplot as plt

from rdkit import Chem
from rdkit.Chem.Fragments import fr_Al_OH


def getElementNames() -> list:
    '''
    Returns the list of (drug-conducive) chemical elements in the order of occurence in the binary vector
    CHNOPS++
    '''
    
    # drugelems = ['B', 'Br', 'C', 'Ca', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']  # all
    
    drugelems = ['Br', 'Cl', 'F', 'I', 'N', 'O', 'S'] # del non-discriminative elements
    
    return drugelems

def getCPEActivity(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Binarizes `CPE.ACTIVITY`
    '''
    
    assert 'CPE.ACTIVITY' in df.columns, "Column `CPE.ACTIVITY` is not present in provided dataframe."
    
    df['cpe'] = df['CPE.ACTIVITY'].str.replace('LOW', '0').str.replace('MODERATE', '1').str.replace('HIGH', '1').astype(int)
    df['ace2'] = df['ACE2.ACTIVITY'].str.replace('LOW', '1').str.replace('MODERATE', '1').str.replace('HIGH', '1').fillna('0')

    return df

def getBinaryColumns(order:int) -> list:
    '''
    Returns exclusively columns with binary entries
    '''
    
    binColumns = getElementNames() + [f'aromRing_{i}' for i in range(order)] + ['cpe', 'ace2'] + ['Label', 'largeMolecule', 'hydroxylGroup']
    
    return binColumns 
    

def getElementsBinary(x:str, pandasFlag:bool=False) -> dict:
    '''
    Extracts available elements from SMILES str and converts it to binary vector over (drug-conducive) chemical elements.
    '''
    
    # available elements
    elems = getElementNames()
    
    # convert
    mol = Chem.MolFromSmiles(str(x))
    inDict = {atom.GetSymbol() for atom in mol.GetAtoms()}
    
    # pandas-like output
    if(pandasFlag):
        return {k:(1 if k in inDict else 0) for k in elems}
    
    return [1 if k in inDict else 0 for k in elems]

def getFormalCharge(x:str, order:int=4, pandasFlag:bool=False) -> dict():
    '''
    Returns if the first `order` rings are aromatic (or not)
    '''
    
    assert isinstance(order, int), "Input `order` must be positive integer."
    
    x = str(x)
    
    m = Chem.MolFromSmiles(x)
    ri = m.GetRingInfo()

    if(pandasFlag):
        return {'charge' : rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges(m)}

    return [rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges(m)]

def getNumberOfAtoms(x:str, pandasFlag:bool=False) -> list:
    '''
    Returns the number of atoms in smiles string
    '''
    
    mol = Chem.MolFromSmiles(str(x))
    
    if(pandasFlag):
        return {'numMolecules' : mol.GetNumAtoms()}
    
    return mol.GetNumAtoms()

def isLargeMolecule(x:str, threshhold:int=30, pandasFlag:bool=False) -> list:
    '''
    Returns if molecule is "large"
    '''
    if(pandasFlag):
        return {'largeMolecule' : int(getNumberOfAtoms(x) >= threshhold)}
    
    return int(getNumberOfAtoms(x) >= threshhold)

def getAromaticRings(x:str, order:int=4, pandasFlag:bool=False) -> dict():
    '''
    Returns if the first `order` rings are aromatic (or not)
    '''
    
    assert isinstance(order, int), "Input `order` must be positive integer."
    
    x = str(x)
    
    m = Chem.MolFromSmiles(x)
    ri = m.GetRingInfo()

    if(pandasFlag):
        return {f'aromRing_{i}' : int(m.GetBondWithIdx(i).GetIsAromatic()) for i in range(order)}

    return [int(m.GetBondWithIdx(i).GetIsAromatic()) for i in range(order)]


def getHydroxylGroup(x:str, pandasFlag=False) -> list:
    '''
    Return hydroxul group.
    '''
    m = Chem.MolFromSmiles(str(x))
    
    
    if(pandasFlag):
        return {'hydroxylGroup' : int(fr_Al_OH(m) > 0)}
    
    return [int(fr_Al_OH(m) > 0)]



def getY(df_path:Path, label:str) -> pd.DataFrame:
    '''
    Return column of target value for `logPow`, `sol`(ubility), `label`
    '''
    
    assert isinstance(label, str) and label in ['logPow', 'sol'], "Target `label` name is either `logPow` or `sol` (solubility)."

    df = pd.read_csv(df_path)
    df = df.rename(columns={'logPow {predicted by ochem.eu/model/535 in Log unit}' : 'logPow',
                            'Aqueous Solubility {predicted by ochem.eu/model/536 in log(mol/L)}' : 'sol'})

    return df[label]



def getProcessedDF(df_path:Path, order:int=4, binaryFlag:bool=False) -> pd.DataFrame:
    '''
    reads in kaggle competition's dataframe, returns processed dataframe with added binary variables
    '''

    # obtain columns
    df_tr      = pd.read_csv(df_path)
    colElem    = pd.DataFrame(df_tr.SMILES.apply(func=getElementsBinary, pandasFlag=True))
    df_elem_tr = pd.json_normalize(df_tr.SMILES.apply(func=getElementsBinary, pandasFlag=True))

    # binarize columns
    df_tr = getCPEActivity(df_tr)
    
    # extended frame
    df_tr      = pd.merge(left=df_tr, right=df_elem_tr, left_index=True, right_index=True)
    df_tr      = pd.merge(left=df_tr, right=pd.json_normalize(df_tr.SMILES.apply(func=isLargeMolecule, pandasFlag=True)), left_index=True, right_index=True)
    df_tr      = pd.merge(left=df_tr, right=pd.json_normalize(df_tr.SMILES.apply(func=getAromaticRings, order=order, pandasFlag=True)), left_index=True, right_index=True)
    df_tr      = pd.merge(left=df_tr, right=pd.json_normalize(df_tr.SMILES.apply(func=getHydroxylGroup, pandasFlag=True)), left_index=True, right_index=True)

    # binary only?
    if(binaryFlag):
        return df_tr[getBinaryColumns(order=order)]
    
    
    return df_tr