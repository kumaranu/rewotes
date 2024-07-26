import pytest
from ase.atoms import Atoms
import pandas as pd
from pathlib import Path
from kumaranu.basisSetSelector import BasisSetSelector


project_root = Path(__file__).resolve().parents[2]


@pytest.fixture
def molecular_structure():
    return Atoms('CO2', positions=[[0.0, 0.0769, 0.0], [1.1797, -0.1620, -0.0], [-1.1797, 0.1043, 0.0]])


@pytest.fixture
def reference_datapoint():
    return Atoms('CO2', positions=[[0.0, 0.076925121247768, 0.0], [1.179724931716919, -0.162000626325607, -0.0], [-1.179724931716919, 0.104306787252426, 0.0]])


@pytest.fixture
def basis_sets():
    return [
        "STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**",
        "6-311G", "6-311G*", "6-311G**", "6-311++G**", "6-311++G(2d,2p)"
    ]


@pytest.fixture
def error_data_and_basis_sets():
    # Define the path to the CSV file
    project_root = Path("/home/kumaranu/Documents/rewotes/kumaranu/tests")
    error_data_file = project_root / 'three_molecules' / 'basis_set_error_data.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(error_data_file)

    # Process the DataFrame to create the error_data dictionary
    error_data = {}
    for _, row in df.iterrows():
        chemical_name = row['chemical_name']
        errors = row[3:].values.astype(float)
        error_data[chemical_name] = errors

    # Extract the basis sets from the column headers
    basis_sets = df.columns[3:].tolist()

    return error_data, basis_sets


def test_select_basis_set_within_tolerance(
        molecular_structure,
        reference_datapoint,
        error_data_and_basis_sets
):
    error_data, basis_sets = error_data_and_basis_sets

    selector = BasisSetSelector(
        molecular_structure,
        reference_datapoint,
        tolerance=1.0,
        error_data=error_data,
        basis_sets=basis_sets,
    )
    selected_basis = selector.select_basis_set()
    assert selected_basis == "3-21G", f"Expected '6-31G' but got {selected_basis}"


def test_select_basis_set_best_available(
        molecular_structure,
        reference_datapoint,
        error_data_and_basis_sets,
):
    error_data, basis_sets = error_data_and_basis_sets
    selector = BasisSetSelector(
        molecular_structure,
        reference_datapoint,
        tolerance=0.01,
        error_data=error_data,
        basis_sets=basis_sets,
    )
    selected_basis = selector.select_basis_set()
    assert selected_basis == "6-31G", f"Expected '6-31G' but got {selected_basis}"


def test_select_basis_set_mismatched_formulas(
        molecular_structure,
        error_data_and_basis_sets,
):
    error_data, basis_sets = error_data_and_basis_sets
    mismatched_reference = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])  # Mismatched chemical formula
    selector = BasisSetSelector(
        molecular_structure,
        mismatched_reference,
        tolerance=1.0,
        error_data=error_data,
        basis_sets=basis_sets,
    )
    with pytest.raises(ValueError) as excinfo:
        selector.select_basis_set()

    # Assert the expected error message
    assert "The chemical formula for the new geometry" in str(excinfo.value)
    assert "do not match." in str(excinfo.value)


def test_select_basis_set_no_tolerance_met(
        molecular_structure,
        reference_datapoint,
        error_data_and_basis_sets,
):
    error_data, basis_sets = error_data_and_basis_sets
    selector = BasisSetSelector(
        molecular_structure,
        reference_datapoint,
        tolerance=0.01,
        error_data=error_data,
        basis_sets=basis_sets,
    )
    selected_basis = selector.select_basis_set()
    assert selected_basis == "6-31G", f"Expected '6-31G' but got {selected_basis}"
