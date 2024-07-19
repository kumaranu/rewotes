import pytest
from kumaranu.basis_set_provider import get_basis_set
from ase.io import read
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


@pytest.fixture()
def setup_test_environment(tmp_path):
    input_geom = read(project_root / "kumaranu" / "tests" / '16.xyz')

    return input_geom


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
