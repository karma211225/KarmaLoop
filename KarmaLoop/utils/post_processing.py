import torch
import argparse
import os
import sys
from tqdm import tqdm
import pdbfixer
import openmm
import warnings
warnings.filterwarnings('ignore')
pwd_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(pwd_dir)
sys.path.append(project_dir)

ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms
# torch.set_num_threads(24)

def openmm_relax(loop_file, loop_file_dir, out_dir, stiffness=1000., tolerance=100, use_gpu=False):
    # try:
    pdb_path = os.path.join(loop_file_dir,loop_file)
    name = rf'{loop_file.split("_pred")[0]}_post.pdb'
    fixer = pdbfixer.PDBFixer(pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    if stiffness > 0:
        stiffness = stiffness * ENERGY / (LENGTH**2)
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        for residue in modeller.topology.residues():
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "C", "CB"]:
                    force.addParticle(
                            atom.index, modeller.positions[atom.index]
                    )
        system.addForce(force)

    tolerance = tolerance * ENERGY
    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getKineticEnergy() + state.getPotentialEnergy()

    with open(rf'{out_dir}/{name}', "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True
        )
    return energy
    # except:
    #     print(loop_file)
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--modeled_loop_dir', type=str, default='/root/KarmaLoop/example/CASP15/test_result/0')
    argparser.add_argument('--output_dir', type=str, default='/root/KarmaLoop/example/CASP15/test_result/post')
    args = argparser.parse_args()
    out_dir = args.output_dir
    # mkdirs
    os.makedirs(out_dir, exist_ok=True) 
    # post processing
    print('########### Start post processing ###########')
    # opmm 
    for file_ in os.listdir(args.modeled_loop_dir):
        if 'pred' in file_:
            try:
                openmm_relax(file_, args.modeled_loop_dir, out_dir,  stiffness=1000, tolerance=100, use_gpu=True)
            except Exception as e:
                print(f'{file_} error due to {e}')