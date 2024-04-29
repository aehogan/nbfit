from io import StringIO
from typing import Union

import h5py
import numpy as np
import openmm
from openff.interchange import Interchange
from openff.interchange.components.potentials import Potential
from openff.interchange.models import PotentialKey
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
    omm_force: Union[openmm.NonbondedForce, openmm.CustomNonbondedForce] = None
    interchange: Interchange = None

    def create_omm_objects(self, forcefield):
        topology: Topology = Topology.from_molecules(self.mols)
        self.interchange = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
        self.omm_system = self.interchange.to_openmm(combine_nonbonded_forces=False)
        omm_integrator = openmm.VerletIntegrator((1 * unit.femtosecond).m_as("picoseconds"))
        self.omm_context = openmm.Context(self.omm_system, omm_integrator,
                                          openmm.Platform.getPlatformByName("Reference"))

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

    def get_pot_keys(self, collection: str = 'DampedExp6810'):
        keys = []
        vals = []
        for system in self.systems:
            for key in list(system.interchange.collections[collection].potentials.keys()):
                if key not in keys:
                    keys.append(key)
                    vals.append(system.interchange.collections[collection].potentials[key])
        return keys, vals

    def calc_error(self):
        energies = self.eval_energies()
        scaler = np.vectorize(lambda x: x if x <= 0 else self.hyperparameters.max_energy_scale * np.arctan(
            x / self.hyperparameters.max_energy_scale))
        energies = scaler(energies)
        error = np.sum(np.square(energies[:, 0] - energies[:, 1]))
        return error

    def fit(self, collection: str = 'DampedExp6810', parameters_to_fit=None):
        if parameters_to_fit is None and collection is 'DampedExp6810':
            parameters_to_fit = ['rho', 'beta']
        elif parameters_to_fit is None and collection is 'vdw':
            parameters_to_fit = ['sigma', 'epsilon']

        keys, vals = self.get_pot_keys(collection=collection)

        # for key, val in zip(keys, vals):
        #    print(key)
        #    print(val)
        #    print()

        error = self.calc_error()

        print(f"\n\nstarting error {error}")

        for key, val in zip(keys, vals):
            val: Potential
            print(key.id)
            for parameter_to_fit in parameters_to_fit:
                print(parameter_to_fit)
                print(val.parameters[parameter_to_fit])
                self.update_params_in_contexts(key, val, collection, parameter_to_fit)
                new_error = self.calc_error()

    def update_params_in_contexts(self, key: PotentialKey, val: Potential, collection: str, parameter_to_fit: str):
        for system in self.systems:
            if key in list(system.interchange.collections[collection].potentials.keys()):
                print(system.interchange.collections[collection].potentials[key].parameters[parameter_to_fit])
                if system.omm_force is None:
                    if collection == 'DampedExp6810':
                        forces = [force for force in system.omm_system.getForces() if
                                  isinstance(force, openmm.CustomNonbondedForce)]
                        system.omm_force = forces[0]
                    elif collection == 'vdw':
                        forces = [force for force in system.omm_system.getForces() if
                                  isinstance(force, openmm.NonbondedForce)]
                        system.omm_force = forces[0]
                for idx, params in enumerate(system.interchange.collections['DampedExp6810'].get_system_parameters()):
                    system.omm_force.setParticleParameters(idx, params)
                system.omm_force.updateParametersInContext(system.omm_context)
