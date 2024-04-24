import pytest
import glob
from openff.toolkit import Molecule, ForceField
from rdkit.Chem import SDMolSupplier

from nbfit.fitting.nbfit import NBFit


def test_methanolx8():
    nbfit = NBFit(forcefield=ForceField("forcefields/openff-2.1.0.offxml", load_plugins=True))
    filenames = ["example_hdf5/NBC10A.hdf5"]
    nbfit.load_hdf5s(filenames)
    nbfit.eval_energies()
