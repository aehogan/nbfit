from io import StringIO

import h5py
import numpy as np
import openmm
from openff.interchange import Interchange
from openff.models.models import DefaultModel
from openff.toolkit import Molecule, ForceField, Topology
from openff.models.types import ArrayQuantity, FloatQuantity
from openff.units import Quantity, unit
from openff.units.openmm import to_openmm, from_openmm
from openmm import System, Context
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from nbfit.utility.atom_conversions import number_to_symbol


class InteractingSystem(DefaultModel):
    mols: list[Molecule]
    ai_energy: list[FloatQuantity["kilocalories_per_mole"]]
    positions: ArrayQuantity["nanometers"]
    omm_system: System = None
    omm_context: Context = None
    interchange: Interchange = None

    def create_omm_objects(self, forcefield):
        topology: Topology = Topology.from_molecules(self.mols)
        self.interchange = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
        self.omm_system = self.interchange.to_openmm(combine_nonbonded_forces=False)
        omm_integrator = openmm.VerletIntegrator((1 * unit.femtosecond).m_as("picoseconds"))
        self.omm_context = openmm.Context(self.omm_system, omm_integrator, openmm.Platform.getPlatformByName("Reference"))

    @staticmethod
    def are_isomorphic(first, other) -> bool:
        if len(first.mols) != len(other.mols):
            return False
        for mol1, mol2 in zip(first.mols, other.mols):
            if not Molecule.are_isomorphic(mol1, mol2):
                return False
        return True


class Hyperparameters(DefaultModel):
    max_energy_scale: float = 10


class NBFit(DefaultModel):
    forcefield: ForceField
    systems: list[InteractingSystem] = []
    hyperparameters: Hyperparameters = Hyperparameters()

    def load_hdf5s(self, filenames: list[str]):
        for filename in filenames:
            file = h5py.File(filename, 'r')
            for key in file.keys():
                dset = file[key]
                n_mols = dset.attrs['n_molecules']
                rdkit_mols = []
                for i in range(n_mols):
                    xyz_stringio = StringIO()
                    xyz_stringio.write(f"{dset['n_atoms'][i]}\n\n")
                    atom_offset = 0
                    for j in range(i):
                        atom_offset += dset['n_atoms'][j]
                    for z, x in zip(dset['atomic_numbers'][atom_offset:atom_offset + dset['n_atoms'][i]],
                                    dset['coordinates'][0][atom_offset:atom_offset + dset['n_atoms'][i]]):
                        xyz_stringio.write(f"{number_to_symbol[z]} {x[0]:f} {x[1]:f} {x[2]:f}\n")
                    mol_from_xyz_block = Chem.rdmolfiles.MolFromXYZBlock(xyz_stringio.getvalue())
                    if dset['n_atoms'][i] > 1:
                        rdDetermineBonds.DetermineBonds(mol_from_xyz_block, charge=int(dset['mol_charges'][i]))
                    mol_from_xyz_block.UpdatePropertyCache()
                    rdkit_mols.append(mol_from_xyz_block)
                openff_mols = [Molecule.from_rdkit(mol) for mol in rdkit_mols]

                positions = dset['coordinates'][:] / 10.0
                energies = [Quantity(energy, unit.kilojoule_per_mole) for energy in dset['ccsd(t)_cbs.energy']]
                system = InteractingSystem(mols=openff_mols, ai_energy=energies, positions=positions)
                system.create_omm_objects(self.forcefield)
                self.systems.append(system)

    def eval_energies(self):
        energies = []
        for system in self.systems:
            for pos_idx in range(system.positions.shape[0]):
                system.omm_context.setPositions(to_openmm(system.positions[pos_idx]))
                omm_state: openmm.State = system.omm_context.getState(getEnergy=True)
                close_energy = from_openmm(omm_state.getPotentialEnergy())

                tmp = [mol.n_atoms for mol in system.mols]
                tmp2 = np.vstack([np.zeros((n_atoms, 3)) + (idx * 100) for idx, n_atoms in enumerate(tmp)])
                tmp3 = Quantity(tmp2, unit.nanometer)

                system.omm_context.setPositions(to_openmm(system.positions[pos_idx] + tmp3))
                omm_state: openmm.State = system.omm_context.getState(getEnergy=True)
                far_energy = from_openmm(omm_state.getPotentialEnergy())
                ai_energy = system.ai_energy[pos_idx]
                energies.append([(close_energy - far_energy).m_as('kilojoule_per_mole'),
                                ai_energy.m_as('kilojoule_per_mole')])
        return energies

    def calc_error(self):
        energies = self.eval_energies()
        scaler = np.vectorize(lambda x: x if x <= 0 else self.hyperparameters.max_energy_scale * np.arctan(x/self.hyperparameters.max_energy_scale))
        energies = scaler(energies)
        error = np.sum(np.square(energies[:, 0] - energies[:, 1]))
        return error

    def fit(self):
        return

    def update_params_in_contexts(self):
        return
