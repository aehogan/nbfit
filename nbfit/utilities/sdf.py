from openff.toolkit import Molecule
from openff.units import Quantity


def load_sdfs(filenames):
    mols = []
    for filename in filenames:
        f = open(filename, "r")
        energy = Quantity(f.readline().strip())
        f.close()
        mol = Molecule.from_file(filename)
        mol = mol.canonical_order_atoms()
        mols.append(mol)
    return mols

