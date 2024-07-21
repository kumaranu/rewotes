import pytest
from kumaranu.basis_set_provider import get_basis_set
from ase.io import read
from pathlib import Path
from ase.calculators.nwchem import NWChem
import os
import shutil

project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment(tmp_path, request):
    # input_geom = read(project_root / "kumaranu" / "tests" / '16.xyz')
    input_geom = read(project_root / "kumaranu" / "tests" / 'molecule_xyz_files' / '04_C1H2O1_geometry_1.xyz')

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


def test_nwchem_ase_calc(setup_test_environment):
    input_ase_obj = setup_test_environment
    # Ignoring my error in the scratch directory assignment for now and cleaning it up afterwards
    input_ase_obj.calc = NWChem(dft=dict(maxiter=2000, xc='B3LYP'), basis='STO-3G')
    err_percent = (input_ase_obj.get_potential_energy()/27.2114 - (-114.38086655385808))/(-114.38086655385808)*100
    print('')
    print(err_percent)
    input_ase_obj.calc = NWChem(dft=dict(maxiter=2000, xc='B3LYP'), basis='3-21G')
    err_percent = (input_ase_obj.get_potential_energy()/27.2114 - (-114.38086655385808))/(-114.38086655385808)*100
    print(err_percent)
    input_ase_obj.calc = NWChem(dft=dict(maxiter=2000, xc='B3LYP'), basis='6-31G')
    err_percent = (input_ase_obj.get_potential_energy()/27.2114 - (-114.38086655385808))/(-114.38086655385808)*100
    print(err_percent)
    input_ase_obj.calc = NWChem(dft=dict(maxiter=2000, xc='B3LYP'), basis='6-31G*')
    err_percent = (input_ase_obj.get_potential_energy()/27.2114 - (-114.38086655385808))/(-114.38086655385808)*100
    print(err_percent)
    # assert input_ase_obj.get_potential_energy() == pytest.approx(-7394.730594653764, abs=1e-5)
    #assert input_ase_obj.get_potential_energy() == pytest.approx(-3115.4232282956423, abs=1e-5)
    err_percent = (input_ase_obj.get_potential_energy()/27.2114 - (-114.38086655385808))/(-114.38086655385808)*100
    assert err_percent == pytest.approx(0.01, abs=1e-2)


def test_get_basis_set_high_precision(setup_test_environment):
    input_ase_obj = setup_test_environment
    basis_set = get_basis_set(input_ase_obj, 0.0, 0.005)
    assert basis_set == "cc-pVTZ"


def test_get_basis_set_medium_precision(setup_test_environment):
    input_ase_obj = setup_test_environment
    basis_set = get_basis_set(input_ase_obj, 0.0, 0.03)
    assert basis_set == "cc-pVDZ"


def test_get_basis_set_low_precision(setup_test_environment):
    input_ase_obj = setup_test_environment
    basis_set = get_basis_set(input_ase_obj, 0.0, 0.1)
    assert basis_set == "STO-3G"
