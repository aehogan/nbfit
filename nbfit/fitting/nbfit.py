import numpy
import openmm
from openff.interchange import Interchange
from openff.models.models import DefaultModel
from openff.toolkit import Molecule, ForceField, Topology
from openff.models.types import ArrayQuantity, FloatQuantity
from openff.units import Quantity, unit
from openff.units.openmm import to_openmm, from_openmm
from openmm import System, Context


class InteractingSystem(DefaultModel):
    mols: list[Molecule] = []
    ai_energy: FloatQuantity["kilocalories_per_mole"] = 0.0 * unit.kilocalories_per_mole

    @staticmethod
    def are_isomorphic(first, other) -> bool:
        if len(first.mols) != len(other.mols):
            return False
        for mol1, mol2 in zip(first.mols, other.mols):
            if not Molecule.are_isomorphic(mol1, mol2):
                return False
        return True


class NBFit(DefaultModel):
    forcefield: ForceField
    systems: list[InteractingSystem] = []
    unique_systems: list[int] = []
    unique_system_map: dict[int, int] = {}
    unique_omm_systems: dict[int, System] = {}
    unique_omm_contexts: dict[int, Context] = {}

    def load_sdfs(self, filenames: list[str]):
        for filename in filenames:
            mols = Molecule.from_file(filename)
            if isinstance(mols, Molecule):
                mols = [mols]
            for idx, mol in enumerate(mols):
                mols[idx] = mol.canonical_order_atoms()
            mols.sort(key=lambda mol: mol.to_inchikey())
            system = InteractingSystem(mols=mols, ai_energy=Quantity(mol.name))
            self.systems.append(system)
        self._update_mol_map()
        self._update_omm_contexts()

    def _update_mol_map(self):
        self.unique_system_map = {}
        for idx, system in enumerate(self.systems):
            for val in self.unique_system_map.values():
                if InteractingSystem.are_isomorphic(system, self.systems[val]):
                    self.unique_system_map[idx] = val
                    break
            if idx not in self.unique_system_map.keys():
                self.unique_system_map[idx] = idx
        self.unique_systems = list(set([val for val in self.unique_system_map.values()]))

    def _update_omm_contexts(self):
        self.unique_omm_systems = {}
        self.unique_omm_contexts = {}
        for unique_index in self.unique_systems:
            system = self.systems[unique_index]
            topology: Topology = Topology.from_molecules(system.mols)
            interchange: Interchange = Interchange.from_smirnoff(force_field=self.forcefield, topology=topology)
            omm_system = interchange.to_openmm(combine_nonbonded_forces=False)
            self.unique_omm_systems[unique_index] = omm_system
            omm_integrator = openmm.VerletIntegrator((1 * unit.femtosecond).m_as("picoseconds"))
            omm_context = openmm.Context(omm_system, omm_integrator, openmm.Platform.getPlatformByName("Reference"))
            self.unique_omm_contexts[unique_index] = omm_context

    def eval_energies(self):
        for idx, system in enumerate(self.systems):
            omm_context = self.unique_omm_contexts[self.unique_system_map[idx]]
            pos = numpy.vstack([mol.conformers[0] for mol in system.mols])
            omm_context.setPositions(to_openmm(pos))
            omm_state: openmm.State = omm_context.getState(getEnergy=True)
            print(from_openmm(omm_state.getPotentialEnergy()))
