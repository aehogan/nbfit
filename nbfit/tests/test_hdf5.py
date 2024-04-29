import pytest
import glob
from openff.toolkit import Molecule, ForceField
from rdkit.Chem import SDMolSupplier

from nbfit.fitting.nbfit import NBFit


def test_nbc10a():
    nbfit = NBFit(forcefield=ForceField("forcefields/openff-2.1.0.offxml", load_plugins=True))
    filenames = ["example_hdf5/NBC10A.hdf5"]
    nbfit.load_hdf5s(filenames)
    # print(nbfit.eval_energies())
    print(nbfit.calc_error())

    nbfit = NBFit(forcefield=ForceField("forcefields/PHAST-H2CNO-2.0.0.offxml", load_plugins=True))
    filenames = ["example_hdf5/NBC10A.hdf5"]
    nbfit.load_hdf5s(filenames)
    # print(nbfit.eval_energies())
    print(nbfit.calc_error())


def test_q_aqua():
    nbfit = NBFit(forcefield=ForceField("forcefields/PHAST-H2CNO-2.0.0.offxml", load_plugins=True))
    filenames = ["example_hdf5/q-AQUA.hdf5"]
    nbfit.load_hdf5s(filenames)
    print(nbfit.eval_energies())
    # print(nbfit.calc_error())


def test_coverage():
    nbfit = NBFit(forcefield=ForceField("forcefields/PHAST-H2CNO-2.0.0.offxml", load_plugins=True))
    filenames = ["example_hdf5/NBC10A.hdf5"]
    nbfit.load_hdf5s(filenames)
    print(nbfit.get_pot_keys(collection='DampedExp6810'))


def test_fit():
    nbfit = NBFit(forcefield=ForceField("forcefields/PHAST-H2CNO-2.0.0.offxml", load_plugins=True))
    filenames = ["example_hdf5/NBC10A.hdf5"]
    nbfit.load_hdf5s(filenames)
    nbfit.fit(collection='DampedExp6810')
