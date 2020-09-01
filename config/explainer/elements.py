from enum import Enum
from rdkit import Chem

class Elements(Enum):
    O = 0
    C = 1
    N = 2
    F = 3
    Cl = 4
    S = 5
    Br = 6
    Si = 7
    Na = 8
    I = 9
    Hg = 10
    B = 11
    K = 12
    P = 13
    Au = 14
    Cr = 15
    Sn = 16
    Ca = 17
    Cd = 18
    Zn = 19
    V = 20
    As = 21
    Li = 22
    Cu = 23
    Co = 24
    Ag = 25
    Se = 26
    Pt = 27
    Al = 28
    Bi = 29
    Sb = 30
    Ba = 31
    Fe = 32
    H = 33
    Ti = 34
    Tl = 35
    Sr = 36
    In = 37
    Dy = 38
    Ni = 39
    Be = 40
    Mg = 41
    Nd = 42
    Pd = 43
    Mn = 44
    Zr = 45
    Pb = 46
    Yb = 47
    Mo = 48
    Ge = 49
    Ru = 50
    Eu = 51
    Sc = 52


def Edges(bond_type):
    if bond_type == Chem.BondType.SINGLE:
        return 0
    elif bond_type == Chem.BondType.DOUBLE:
        return 1
    elif bond_type == Chem.BondType.AROMATIC:
        return 2
    elif bond_type == Chem.BondType.TRIPLE:
        return 3
    else:
        raise Exception("No bond type found")


def EdgesToRDKit(bond_type):
    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.AROMATIC
    elif bond_type == 3:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")
