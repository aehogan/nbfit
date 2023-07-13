from openff.interchange import Interchange
from openff.models.models import DefaultModel
from openff.toolkit import Molecule, ForceField, Topology
from openff.models.types import ArrayQuantity
from openff.units import Quantity, unit
from openmm import System, Context


class NBFit(DefaultModel):
    mols: list[Molecule] = []
    ai_energies: ArrayQuantity["kilocalories_per_mole"] = []
    mol_map: dict[int, int] = {}
    unique_mols: list[int] = []
    forcefield: ForceField
    omm_systems: dict[int, System] = {}
    omm_contexts: dict[int, Context] = {}

    def load_sdfs(self, filenames: list[str]):
        for filename in filenames:
            mol = Molecule.from_file(filename)
            mol = mol.canonical_order_atoms()
            self.ai_energies.append(Quantity(mol.name))
            self.mols.append(mol)
        self._update_mol_map()

    def _update_mol_map(self):
        self.mol_map = {}
        for idx, mol in enumerate(self.mols):
            for val in self.mol_map.values():
                if Molecule.are_isomorphic(mol, self.mols[val])[0]:
                    self.mol_map[idx] = val
                    break
            if idx not in self.mol_map.keys():
                self.mol_map[idx] = idx
        self.unique_mols = list(set([val for val in self.mol_map.values()]))

    def _update_omm_contexts(self):
        self.omm_systems = {}
        self.omm_contexts = {}
        for unique_index in self.unique_mols:
            mol = self.mols[unique_index]
            topology: Topology = mol.to_topology()
            topology.box_vectors = Quantity([10, 10, 10], unit.nanometer)
            out = Interchange.from_smirnoff(force_field=self.forcefield, topology=topology)
            system = out.to_openmm(combine_nonbonded_forces=False)
            self.omm_systems[unique_index] = system


    def eval_energies(self):
        self._update_omm_contexts()
