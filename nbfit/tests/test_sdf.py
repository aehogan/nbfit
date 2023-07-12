import pytest
import glob
from nbfit.utilities.sdf import load_sdfs
from openff.toolkit import Molecule


def test_methanolx8():
    filenames = glob.glob("example_sdfs/S66x8/*.sdf")
    mols = load_sdfs(filenames)
    for mol in mols:
        print(Molecule.are_isomorphic(mols[0], mol, return_atom_map=True))
