import pytest
import glob
from openff.toolkit import Molecule, ForceField
from rdkit.Chem import SDMolSupplier

from nbfit.fitting.nbfit import NBFit


def test_methanolx8():
    nbfit = NBFit(forcefield=ForceField("forcefields/openff-2.1.0.offxml", load_plugins=True))
    filenames = glob.glob("example_sdfs/S66x8/*.sdf")
    nbfit.load_sdfs(filenames)
    print(nbfit.mols)
    print(nbfit.ai_energies)
    print(nbfit.mol_map)
    print(nbfit.eval_energies())
