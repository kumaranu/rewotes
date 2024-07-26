import glob
import os

import numpy as np
import pytest
import shutil
import pandas as pd
from ase.io import read
from pathlib import Path
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.calculators.nwchem import NWChem
from ase.thermochemistry import IdealGasThermo

project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment1(request):
    input_geom1 = read(project_root / "kumaranu" / "tests" / 'molecule_xyz_files' / '04_C1H2O1_geometry_first.xyz')
    return cleanup_test_environment(request, input_geom1)


@pytest.fixture()
def setup_test_environment2(request):
    input_geom2 = read(project_root / "kumaranu" / "tests" / 'molecule_xyz_files' / '04_C1H2O1_geometry_last.xyz')
    return cleanup_test_environment(request, input_geom2)


def cleanup_test_environment(request, input_geom):
    def cleanup():
        files_and_dirs = [
            project_root / "nwchem.nwi",
            project_root / "nwchem.nwo",
            project_root / "nwchem"
        ]
        for item in files_and_dirs:
            if os.path.isfile(item):
                print(f"Removing file: {item}")
                os.remove(item)
            elif os.path.isdir(item):
                print(f"Removing directory: {item}")
                shutil.rmtree(item)
            else:
                print(f"{item} does not exist.")

    request.addfinalizer(cleanup)
    return input_geom


def calculate_energy(input_ase_obj1, basis, vib_analysis=False):
    # This function calculates Gibbs free energy but does not use it anywhere.
    input_ase_obj1.calc = NWChem(
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
        basis=basis,
    )
    if vib_analysis:
        dyn = QuasiNewton(input_ase_obj1)
        dyn.run(fmax=0.01)
        energy1 = input_ase_obj1.get_potential_energy()
        vib = Vibrations(input_ase_obj1)
        vib.run()
        vib_energies = vib.get_energies()
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry='nonlinear',
            potentialenergy=energy1,
            atoms=input_ase_obj1,
            symmetrynumber=1,
            spin=0,
        )
        G = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.)
        return G

    energy1 = input_ase_obj1.get_potential_energy()

    return energy1


# Load the error data from the CSV file
def load_error_data(csv_file):
    df = pd.read_csv(csv_file)
    error_data = {}

    for _, row in df.iterrows():
        chemical_name = row['chemical_name']
        errors = row[3:].values.astype(float)  # Get error values and convert to float
        error_data[chemical_name] = errors

    return error_data, df.columns[3:].tolist()  # return error data and basis sets


def select_basis_set(new_geometry, known_geometry, tolerance):
    # Load the error data
    error_data, basis_sets = load_error_data(
        f'{str(project_root)}/kumaranu/tests/molecule_xyz_files/basis_set_error_data.csv',
    )
    new_formula = str(new_geometry.symbols)
    known_formula = str(known_geometry.symbols)
    if known_formula != new_formula:
        raise ValueError(f"The chemical formula for the new geometry and the reference do not match.")

    errors = np.abs(np.array(error_data[known_formula]))
    below_tolerance = errors <= tolerance

    if any(below_tolerance):
        selected_index = np.argmax(below_tolerance)
        selected_basis = basis_sets[selected_index][:-14]
        return selected_basis
    else:
        best_index = np.argmin(errors)
        best_basis = basis_sets[best_index][:-14]
        print(f"Warning: No basis set can satisfy the tolerance of {tolerance}. "
              f"Using the best available basis set, {best_basis}.")
        return best_basis


@pytest.mark.skip()
def test_select_basis_set_low_tol():
    new_geometry = read(project_root / 'kumaranu/tests/molecule_xyz_files/03_C1O2_geometry_last.xyz')
    reference_geometry = read(project_root / 'kumaranu/tests/molecule_xyz_files/03_C1O2_geometry_first.xyz')
    tolerance = 4.0
    assert select_basis_set(new_geometry, reference_geometry, tolerance) == 'STO-3G'


@pytest.mark.skip()
def test_select_basis_set_medium_tol():
    new_geometry = read(project_root / 'kumaranu/tests/molecule_xyz_files/03_C1O2_geometry_last.xyz')
    reference_geometry = read(project_root / 'kumaranu/tests/molecule_xyz_files/03_C1O2_geometry_first.xyz')
    tolerance = 0.1
    assert select_basis_set(new_geometry, reference_geometry, tolerance) == '6-31G'


@pytest.mark.skip()
def test_select_basis_set_high_tol():
    new_geometry = read(project_root / 'kumaranu/tests/molecule_xyz_files/03_C1O2_geometry_last.xyz')
    reference_geometry = read(project_root / 'kumaranu/tests/molecule_xyz_files/03_C1O2_geometry_first.xyz')
    tolerance = 0.01
    assert select_basis_set(new_geometry, reference_geometry, tolerance) == '6-31G'


@pytest.mark.skip()
def test_collect_and_store_data(
        output_file=f'{str(project_root)}/kumaranu/tests/molecule_xyz_files/basis_set_error_data.csv',
):
    basis_sets = [
        "STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**",
        "6-311G", "6-311G*", "6-311G**", "6-311++G**", "6-311++G(2d,2p)",
    ]

    data = []
    mol_list = glob.glob(f'{str(project_root)}/kumaranu/tests/molecule_xyz_files/*_first.xyz')

    for mol in mol_list:
        input_ase_obj1 = read(mol)
        ref_energy = float(list(input_ase_obj1.info)[1])
        row_data = {
             'chemical_name': input_ase_obj1.symbols,
             'chemical_symbols': input_ase_obj1.get_chemical_symbols(),
             'geometry': input_ase_obj1.positions.tolist(),
             }

        for basis in basis_sets:
            try:
                energy = calculate_energy(input_ase_obj1, basis)
                err_percent = (abs(energy / 27.2114 - ref_energy) / ref_energy) * 100
                row_data[basis + '-error-percent'] = err_percent
            except Exception as e:
                print(f"An error occurred with basis set {basis}: {e}")
                row_data[basis + 'error-percent'] = None
        data.append(row_data)
        print(f'Done with {mol}.')

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f'Data has been saved to {output_file}')


@pytest.mark.skip()
def test_nwchem_ase_calc_raw(setup_test_environment1):
    input_ase_obj = setup_test_environment1
    input_ase_obj.calc = NWChem(dft=dict(maxiter=2000, xc='B3LYP'), basis='6-31+G*')
    calculated_energy = input_ase_obj.get_potential_energy()
    assert calculated_energy == pytest.approx(-3115.4232282956423, abs=1e-5)
