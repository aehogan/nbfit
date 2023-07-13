from openff.models.models import DefaultModel
from openff.toolkit import Molecule, ForceField
from openff.models.types import ArrayQuantity
from openff.units import Quantity
from openmm import System, Context


class NBFit(DefaultModel):
    mols: list[Molecule] = []
    ai_energies: ArrayQuantity["kilocalories_per_mole"] = []
    mol_map: dict[int, int] = {}
    forcefield: ForceField
    omm_systems: list[System] = []
    omm_contexts: list[Context] = []

    def load_sdfs(self, filenames: list[str]):
        for filename in filenames:
            mol = Molecule.from_file(filename)
            mol = mol.canonical_order_atoms()
            self.ai_energies.append(Quantity(mol.name))
            self.mols.append(mol)
