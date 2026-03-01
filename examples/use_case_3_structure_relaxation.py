#!/usr/bin/env python3
"""
HighFold-MeD Use Case 3: Structure Relaxation with OpenMM

This script performs structure relaxation/refinement of predicted cyclic peptide
structures using OpenMM molecular dynamics. Based on relaxed.py.

Usage:
    python use_case_3_structure_relaxation.py --pdb examples/data/structures/1.pdb --output relaxed_1.pdb
    python use_case_3_structure_relaxation.py --pdb_dir examples/data/structures/ --output_dir relaxed_structures/
"""

import argparse
import os
import sys
from pathlib import Path

def check_openmm_available():
    """Check if OpenMM is available for structure relaxation."""
    try:
        from openmm import LangevinIntegrator, CustomExternalForce
        from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PDBFile, Simulation, HBonds, NoCutoff
        from openmm.unit import kelvin, picosecond, kilocalories_per_mole
        return True
    except ImportError as e:
        print(f"OpenMM not available: {e}")
        print("Install OpenMM for structure relaxation: mamba install -c conda-forge openmm")
        return False


def create_amber_files_from_pdb(pdb_file, work_dir):
    """
    Create AMBER topology and coordinate files from PDB using tleap.

    Args:
        pdb_file (str): Input PDB file path
        work_dir (str): Working directory for intermediate files

    Returns:
        tuple: (topology_file, coordinate_file) paths
    """

    print(f"Creating AMBER files for: {pdb_file}")

    # Copy PDB to working directory
    import shutil
    work_pdb = os.path.join(work_dir, "input.pdb")
    shutil.copy2(pdb_file, work_pdb)

    # Create tleap input file
    leap_input = os.path.join(work_dir, "leap.in")
    with open(leap_input, 'w') as f:
        f.write("""source leaprc.protein.ff14SB
source leaprc.gaff
source leaprc.water.tip3p

# Load PDB
mol = loadpdb input.pdb

# Save topology and coordinates
saveamberparm mol system.prmtop system.inpcrd

# Quit
quit
""")

    # Run tleap
    os.chdir(work_dir)
    result = os.system('tleap -f leap.in > leap.log 2>&1')

    if result != 0:
        print("Warning: tleap may have encountered issues. Check leap.log for details.")

    topology_file = os.path.join(work_dir, "system.prmtop")
    coordinate_file = os.path.join(work_dir, "system.inpcrd")

    return topology_file, coordinate_file


def relax_structure_openmm(pdb_file, output_file, restraint_force=20000.0, tolerance=2.39):
    """
    Relax protein structure using OpenMM with backbone restraints.

    Args:
        pdb_file (str): Input PDB file
        output_file (str): Output relaxed PDB file
        restraint_force (float): Force constant for position restraints (kJ/mol/nm²)
        tolerance (float): Energy tolerance for minimization (kcal/mol)

    Returns:
        dict: Relaxation results
    """

    if not check_openmm_available():
        return {'status': 'error', 'message': 'OpenMM not available'}

    from openmm import LangevinIntegrator, CustomExternalForce
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PDBFile, Simulation, HBonds, NoCutoff
    from openmm.unit import kelvin, picosecond, kilocalories_per_mole

    print(f"Relaxing structure: {pdb_file}")

    # Create working directory
    work_dir = os.path.splitext(output_file)[0] + "_work"
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Create AMBER files from PDB
        topology_file, coordinate_file = create_amber_files_from_pdb(pdb_file, work_dir)

        if not os.path.exists(topology_file) or not os.path.exists(coordinate_file):
            return {'status': 'error', 'message': 'Failed to create AMBER files'}

        # Load topology and coordinates
        prmtop = AmberPrmtopFile(topology_file)
        inpcrd = AmberInpcrdFile(coordinate_file)

        # Create system
        system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds)

        # Add position restraints for backbone atoms
        force = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", restraint_force)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        # Apply restraints to backbone atoms (CA, C, N, O)
        restrained_atoms = 0
        for i, atom in enumerate(prmtop.topology.atoms()):
            if atom.name in ["CA", "C", "N", "O"]:
                force.addParticle(i, inpcrd.positions[i])
                restrained_atoms += 1

        system.addForce(force)

        print(f"Applied position restraints to {restrained_atoms} backbone atoms")

        # Create integrator and simulation
        integrator = LangevinIntegrator(0, 0.01, 0.0)  # T=0 for minimization
        simulation = Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)

        # Get initial energy
        initial_state = simulation.context.getState(getEnergy=True)
        initial_energy = initial_state.getPotentialEnergy()

        print(f"Initial potential energy: {initial_energy}")

        # Energy minimization
        print("Performing energy minimization...")
        simulation.minimizeEnergy(tolerance=tolerance*kilocalories_per_mole)

        # Get final energy
        final_state = simulation.context.getState(getEnergy=True)
        final_energy = final_state.getPotentialEnergy()

        print(f"Final potential energy: {final_energy}")
        print(f"Energy change: {final_energy - initial_energy}")

        # Get relaxed positions and save
        positions = simulation.context.getState(getPositions=True).getPositions()

        with open(output_file, 'w') as f:
            PDBFile.writeFile(prmtop.topology, positions, f)

        print(f"Relaxed structure saved to: {output_file}")

        return {
            'status': 'success',
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': final_energy - initial_energy,
            'restrained_atoms': restrained_atoms,
            'output_file': output_file
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def relax_structure_demo(pdb_file, output_file):
    """
    Demo version of structure relaxation (when OpenMM is not available).

    Args:
        pdb_file (str): Input PDB file
        output_file (str): Output file path

    Returns:
        dict: Demo results
    """

    print(f"DEMO: Structure relaxation for {pdb_file}")
    print("This would perform energy minimization with backbone restraints")

    # Copy input to output for demo
    import shutil
    shutil.copy2(pdb_file, output_file)

    # Read PDB to get some stats
    atom_count = 0
    residue_count = 0

    with open(pdb_file, 'r') as f:
        residues = set()
        for line in f:
            if line.startswith('ATOM'):
                atom_count += 1
                residues.add(line[22:26].strip())
        residue_count = len(residues)

    print(f"Structure contains: {residue_count} residues, {atom_count} atoms")
    print(f"Demo output saved to: {output_file}")

    return {
        'status': 'demo_success',
        'residue_count': residue_count,
        'atom_count': atom_count,
        'output_file': output_file,
        'message': 'Demo run - install OpenMM for actual relaxation'
    }


def main():
    parser = argparse.ArgumentParser(description="Relax cyclic peptide structures using OpenMM")

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pdb', type=str, help='Single PDB file to relax')
    group.add_argument('--pdb_dir', type=str, help='Directory containing PDB files')

    # Output options
    parser.add_argument('--output', type=str, help='Output PDB file (for single file)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for batch processing)')
    parser.add_argument('--suffix', type=str, default='_relaxed', help='Suffix for output files')

    # Relaxation parameters
    parser.add_argument('--restraint_force', type=float, default=20000.0,
                       help='Force constant for position restraints (kJ/mol/nm²)')
    parser.add_argument('--tolerance', type=float, default=2.39,
                       help='Energy tolerance for minimization (kcal/mol)')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (without OpenMM)')

    args = parser.parse_args()

    print("HighFold-MeD Structure Relaxation")
    print("=" * 40)

    # Check OpenMM availability
    openmm_available = check_openmm_available()
    if not openmm_available and not args.demo:
        print("OpenMM not available. Use --demo flag to run demo version.")
        return 1

    use_demo = args.demo or not openmm_available

    if use_demo:
        print("Running in DEMO mode")

    # Process single file or batch
    results = []

    if args.pdb:
        # Single file processing
        if not os.path.exists(args.pdb):
            print(f"Error: PDB file not found: {args.pdb}")
            return 1

        output_file = args.output or (
            os.path.splitext(args.pdb)[0] + args.suffix + ".pdb"
        )

        if use_demo:
            result = relax_structure_demo(args.pdb, output_file)
        else:
            result = relax_structure_openmm(
                args.pdb, output_file, args.restraint_force, args.tolerance
            )

        results.append(result)

    else:
        # Batch processing
        pdb_dir = args.pdb_dir
        output_dir = args.output_dir or (pdb_dir.rstrip('/') + '_relaxed')

        if not os.path.exists(pdb_dir):
            print(f"Error: PDB directory not found: {pdb_dir}")
            return 1

        os.makedirs(output_dir, exist_ok=True)

        # Find PDB files
        pdb_files = [f for f in os.listdir(pdb_dir) if f.lower().endswith('.pdb')]

        if not pdb_files:
            print(f"No PDB files found in: {pdb_dir}")
            return 1

        print(f"Found {len(pdb_files)} PDB files to process")

        for pdb_file in pdb_files:
            input_path = os.path.join(pdb_dir, pdb_file)
            output_file = os.path.splitext(pdb_file)[0] + args.suffix + ".pdb"
            output_path = os.path.join(output_dir, output_file)

            print(f"\nProcessing: {pdb_file}")

            if use_demo:
                result = relax_structure_demo(input_path, output_path)
            else:
                result = relax_structure_openmm(
                    input_path, output_path, args.restraint_force, args.tolerance
                )

            result['input_file'] = pdb_file
            results.append(result)

    # Print summary
    print("\n" + "=" * 40)
    print("RELAXATION SUMMARY")
    print("=" * 40)

    successful = [r for r in results if 'success' in r['status']]
    errors = [r for r in results if r['status'] == 'error']

    print(f"Total structures: {len(results)}")
    print(f"Successfully relaxed: {len(successful)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for result in errors:
            print(f"  - {result.get('input_file', 'Unknown')}: {result.get('message', 'Unknown error')}")

    if successful and not use_demo:
        print(f"\nEnergy changes:")
        for result in successful:
            energy_change = result.get('energy_change', 'N/A')
            print(f"  - {result.get('input_file', 'Unknown')}: {energy_change}")

    return 0


if __name__ == "__main__":
    exit(main())