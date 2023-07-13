import pytest
import glob
from openff.toolkit import Molecule, ForceField

from nbfit.fitting.nbfit import NBFit


def test_methanolx8():
    filenames = glob.glob("example_sdfs/S66x8/*.sdf")
    nbfit = NBFit(forcefield=ForceField("forcefields/openff-2.1.0.offxml"))
    nbfit.load_sdfs(filenames)
    print(nbfit.mols)
    print(nbfit.ai_energies)
    for mol in nbfit.mols:
        print(Molecule.are_isomorphic(nbfit.mols[0], mol, return_atom_map=True))
