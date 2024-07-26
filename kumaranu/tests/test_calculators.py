import pytest
from ase.atoms import Atoms
from ase.calculators.nwchem import NWChem
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from kumaranu.calculators import EnergyCalculator


@pytest.fixture
def dummy_atoms():
    # Create a dummy H2 molecule
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])


def test_calculate_energy_no_vib(dummy_atoms):
    # Set up the calculator
    dummy_atoms.calc = NWChem(
        dft=dict(
            maxiter=2000,
            xc='B3LYP',
            convergence={
                'energy': 1e-9,
                'density': 1e-7,
                'gradient': 5e-6,
                'hl_tol': 0.01,
            }
        ),
        basis='STO-3G',
    )

    # Call the method without vibrational analysis
    energy = EnergyCalculator.calculate_energy(
        dummy_atoms,
        'STO-3G',
        vib_analysis=False,
    )

    # Assert the potential energy is returned
    assert energy == pytest.approx(-31.712646065466686, abs='1e-6')


def test_calculate_energy_with_vib(dummy_atoms):
    # Set up the calculator
    dummy_atoms.calc = NWChem(
        dft=dict(
            maxiter=2000,
            xc='B3LYP',
            convergence={
                'energy': 1e-9,
                'density': 1e-7,
                'gradient': 5e-6,
                'hl_tol': 0.01,
            }
        ),
        basis='STO-3G',
    )

    # Call the method with vibrational analysis
    gibbs_free_energy = EnergyCalculator.calculate_energy(
        dummy_atoms,
        'STO-3G',
        vib_analysis=True,
        geometry='linear',
        symmetrynumber=1,
    )

    # Assert the Gibbs free energy
    assert gibbs_free_energy == pytest.approx(-31.72846909650759, abs='1e-6')
