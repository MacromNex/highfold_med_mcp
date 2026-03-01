#!/usr/bin/env python3
"""
Script: relax_structure.py
Description: Structure relaxation of cyclic peptides using OpenMM molecular dynamics

Original Use Case: examples/use_case_3_structure_relaxation.py
Dependencies Removed: Simplified OpenMM usage, inlined AMBER file creation logic

Usage:
    python scripts/relax_structure.py --input <pdb_file> --output <output_file>

Example:
    python scripts/relax_structure.py --input examples/data/structures/1.pdb --output results/relaxed.pdb --demo
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import tempfile

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "demo_mode": True,  # Default to demo for MCP
    "restraint_force": 20000.0,  # kJ/mol/nm²
    "tolerance": 2.39,  # kcal/mol
    "backbone_restraints": True,
    "energy_minimization": True,
    "output_format": "pdb"
}

# ==============================================================================
# Utility Functions
# ==============================================================================
def check_openmm_available() -> bool:
    """Check if OpenMM is available for structure relaxation."""
    try:
        from openmm import LangevinIntegrator, CustomExternalForce
        from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PDBFile, Simulation, HBonds, NoCutoff
        from openmm.unit import kelvin, picosecond, kilocalories_per_mole
        return True
    except ImportError:
        return False

def parse_pdb_info(pdb_file: Path) -> Dict[str, Any]:
    """Extract basic information from PDB file."""
    info = {
        'atoms': 0,
        'residues': set(),
        'chains': set(),
        'has_cyclic_structure': False
    }

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    info['atoms'] += 1
                    info['residues'].add(line[22:26].strip())
                    info['chains'].add(line[21])

                # Look for CONECT records indicating cyclic structure
                elif line.startswith('CONECT'):
                    info['has_cyclic_structure'] = True

        info['residue_count'] = len(info['residues'])
        info['chain_count'] = len(info['chains'])

    except Exception as e:
        print(f"Warning: Could not parse PDB file {pdb_file}: {e}")

    return info

def create_amber_files(pdb_file: Path, work_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Create AMBER topology and coordinate files from PDB using tleap."""
    print(f"Creating AMBER files for: {pdb_file}")

    # Copy PDB to working directory
    work_pdb = work_dir / "input.pdb"
    shutil.copy2(pdb_file, work_pdb)

    # Create tleap input file
    leap_input = work_dir / "leap.in"
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
    original_cwd = os.getcwd()
    os.chdir(work_dir)

    try:
        result = os.system('tleap -f leap.in > leap.log 2>&1')

        topology_file = work_dir / "system.prmtop"
        coordinate_file = work_dir / "system.inpcrd"

        if result != 0 or not topology_file.exists() or not coordinate_file.exists():
            print("Warning: tleap encountered issues. Check leap.log for details.")
            return None, None

        return topology_file, coordinate_file

    except Exception as e:
        print(f"Error running tleap: {e}")
        return None, None

    finally:
        os.chdir(original_cwd)

def relax_with_openmm(pdb_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform structure relaxation using OpenMM."""
    if not check_openmm_available():
        return {'status': 'error', 'message': 'OpenMM not available'}

    from openmm import LangevinIntegrator, CustomExternalForce
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PDBFile, Simulation, HBonds, NoCutoff
    from openmm.unit import kelvin, picosecond, kilocalories_per_mole

    print(f"Relaxing structure: {pdb_file}")

    # Create working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)

        try:
            # Create AMBER files from PDB
            topology_file, coordinate_file = create_amber_files(pdb_file, work_dir)

            if not topology_file or not coordinate_file:
                return {'status': 'error', 'message': 'Failed to create AMBER files'}

            # Load topology and coordinates
            prmtop = AmberPrmtopFile(str(topology_file))
            inpcrd = AmberInpcrdFile(str(coordinate_file))

            # Create system
            system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds)

            # Add position restraints for backbone atoms
            if config.get('backbone_restraints', True):
                force = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
                force.addGlobalParameter("k", config.get('restraint_force', DEFAULT_CONFIG['restraint_force']))
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
            else:
                restrained_atoms = 0

            # Create integrator and simulation
            integrator = LangevinIntegrator(0, 0.01, 0.0)  # T=0 for minimization
            simulation = Simulation(prmtop.topology, system, integrator)
            simulation.context.setPositions(inpcrd.positions)

            # Get initial energy
            initial_state = simulation.context.getState(getEnergy=True)
            initial_energy = initial_state.getPotentialEnergy()

            print(f"Initial potential energy: {initial_energy}")

            # Energy minimization
            if config.get('energy_minimization', True):
                print("Performing energy minimization...")
                tolerance = config.get('tolerance', DEFAULT_CONFIG['tolerance'])
                simulation.minimizeEnergy(tolerance=tolerance*kilocalories_per_mole)

                # Get final energy
                final_state = simulation.context.getState(getEnergy=True)
                final_energy = final_state.getPotentialEnergy()

                print(f"Final potential energy: {final_energy}")
                print(f"Energy change: {final_energy - initial_energy}")
            else:
                final_energy = initial_energy

            # Get relaxed positions and save
            positions = simulation.context.getState(getPositions=True).getPositions()

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                PDBFile.writeFile(prmtop.topology, positions, f)

            print(f"Relaxed structure saved to: {output_file}")

            return {
                'status': 'success',
                'initial_energy': str(initial_energy),
                'final_energy': str(final_energy),
                'energy_change': str(final_energy - initial_energy),
                'restrained_atoms': restrained_atoms,
                'output_file': str(output_file)
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

def relax_demo_mode(pdb_file: Path, output_file: Path) -> Dict[str, Any]:
    """Demo version of structure relaxation (when OpenMM is not available)."""
    print(f"DEMO: Structure relaxation for {pdb_file}")
    print("This would perform energy minimization with backbone restraints")

    # Parse PDB to get statistics
    pdb_info = parse_pdb_info(pdb_file)

    # Copy input to output for demo
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdb_file, output_file)

    print(f"Structure contains: {pdb_info['residue_count']} residues, {pdb_info['atoms']} atoms")
    print(f"Demo output saved to: {output_file}")

    return {
        'status': 'demo_success',
        'residue_count': pdb_info['residue_count'],
        'atom_count': pdb_info['atoms'],
        'has_cyclic_structure': pdb_info['has_cyclic_structure'],
        'output_file': str(output_file),
        'message': 'Demo run - install OpenMM for actual relaxation'
    }

# ==============================================================================
# Core Function
# ==============================================================================
def run_relax_structure(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for cyclic peptide structure relaxation.

    Args:
        input_file: Path to input PDB file
        output_file: Path to save relaxed structure (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Relaxation results
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_relax_structure("input.pdb", "relaxed.pdb")
        >>> print(result['result']['status'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not str(input_file).lower().endswith('.pdb'):
        raise ValueError(f"Input file must be PDB format: {input_file}")

    # Generate output filename if not provided
    if not output_file:
        output_file = input_file.with_name(f"{input_file.stem}_relaxed.pdb")
    else:
        output_file = Path(output_file)

    print(f"Input PDB: {input_file}")
    print(f"Output PDB: {output_file}")

    # Check OpenMM availability
    openmm_available = check_openmm_available()
    use_demo = config.get('demo_mode', DEFAULT_CONFIG['demo_mode']) or not openmm_available

    if use_demo:
        print("Running in demo mode")
        result_data = relax_demo_mode(input_file, output_file)
    else:
        print("Running OpenMM structure relaxation")
        result_data = relax_with_openmm(input_file, output_file, config)

    return {
        "result": result_data,
        "output_file": str(output_file),
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "openmm_available": openmm_available,
            "timestamp": pd.Timestamp.now().isoformat() if 'pd' in globals() else "unknown"
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                       help='Input PDB file path')
    parser.add_argument('--output', '-o',
                       help='Output PDB file path (default: <input>_relaxed.pdb)')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON format)')

    # Relaxation parameters
    parser.add_argument('--demo', action='store_true',
                       help='Force demo mode (without OpenMM)')
    parser.add_argument('--restraint_force', type=float,
                       default=DEFAULT_CONFIG['restraint_force'],
                       help=f'Position restraint force constant (default: {DEFAULT_CONFIG["restraint_force"]})')
    parser.add_argument('--tolerance', type=float,
                       default=DEFAULT_CONFIG['tolerance'],
                       help=f'Energy minimization tolerance (default: {DEFAULT_CONFIG["tolerance"]})')
    parser.add_argument('--no_restraints', action='store_true',
                       help='Skip backbone position restraints')
    parser.add_argument('--no_minimization', action='store_true',
                       help='Skip energy minimization')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    config.update({
        'demo_mode': args.demo,
        'restraint_force': args.restraint_force,
        'tolerance': args.tolerance,
        'backbone_restraints': not args.no_restraints,
        'energy_minimization': not args.no_minimization
    })

    # Run structure relaxation
    try:
        result = run_relax_structure(
            input_file=args.input,
            output_file=args.output,
            config=config
        )

        print(f"\nRelaxation completed!")
        print(f"Status: {result['result']['status']}")
        print(f"Output file: {result['output_file']}")

        if 'energy_change' in result['result']:
            print(f"Energy change: {result['result']['energy_change']}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())