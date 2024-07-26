import pytest
from ase.atoms import Atoms
from ase.build import molecule
from kumaranu.basisSetProvider import BasisSetProvider
from kumaranu.dataCollector import DataCollector
from kumaranu.basisSetSelector import BasisSetSelector

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


# Mock data and functions
@pytest.fixture()
def mock_load_error_data(*args, **kwargs):
    # This mock function provides controlled test data
    return {
        'CO2': [1.0, 2.0, 3.0],  # Example error data for CO2
        'H2': [0.1, 0.2, 0.3],  # Example error data for H2
    }, ["STO-3G", "3-21G", "6-31G"]


@pytest.fixture
def setup_basis_set_provider():
    project_root = "/mock/path"
    tolerance = 0.5
    return BasisSetProvider(project_root, tolerance)


@pytest.fixture
def mock_data_collector(monkeypatch):
    # Replace DataCollector's load_error_data method with a mock
    monkeypatch.setattr(DataCollector, 'load_error_data', mock_load_error_data)


def test_get_basis_set_within_tolerance(
        setup_basis_set_provider,
        mock_load_error_data,
):
    mol = Atoms('CO2', positions=[[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    ref = Atoms('CO2', positions=[[0, 0, 0], [1, 1, 1], [-1, -1, -1]])

    tolerance = 0.5
    basisProviderObject = BasisSetProvider(
        project_root,
        tolerance,
        # error_data_file=str(project_root / 'kumaranu/tests/molecule_xyz_files/basis_set_error_data.csv'),
        files_dir=str(project_root / 'kumaranu/tests/three_molecules'),
        recalculate_errors=True,
    )
    selected_basis = basisProviderObject.get_basis_set(mol, ref)

    assert selected_basis == '3-21G'


def test_get_basis_set_mismatched_formulas(
        setup_basis_set_provider,
        mock_load_error_data,
):
    mol = Atoms('CO2', positions=[[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    ref = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])

    tolerance = 0.5
    provider = BasisSetProvider(
        project_root,
        tolerance,
        error_data_file=str(project_root / 'kumaranu/tests/molecule_xyz_files/basis_set_error_data.csv'),
    )
    with pytest.raises(ValueError,
                       match="The chemical formula for the new geometry \(CO2\) and the reference \(H2\) do not match."):
        provider.get_basis_set(mol, ref)
