from io import StringIO

import h5py
import numpy
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

    def create_omm_objects(self, forcefield):
        topology: Topology = Topology.from_molecules(self.mols)
        interchange: Interchange = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
        self.omm_system = interchange.to_openmm(combine_nonbonded_forces=False)
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


class NBFit(DefaultModel):
    forcefield: ForceField
    systems: list[InteractingSystem] = []

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
        for system in self.systems:
            for pos_idx in range(system.positions.shape[0]):
                # TODO get energy far away and subtract off
                system.omm_context.setPositions(to_openmm(system.positions[pos_idx]))
                omm_state: openmm.State = system.omm_context.getState(getEnergy=True)
                print(from_openmm(omm_state.getPotentialEnergy()), system.ai_energy[pos_idx])

    def calc_error(self):
        #fit_energy = system.fit_energy <= 0. ? system.fit_energy: parameters.max_energy * atan(
        #system.fit_energy / parameters.max_energy);
        #calc_energy = system.total_energy <= 0. ? system.total_energy: parameters.max_energy * atan(
        #system.total_energy / parameters.max_energy);
        #error += pow(calc_energy - fit_energy, 2);
        return 0.0

    def update_params_in_contexts(self):
        return
