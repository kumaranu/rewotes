import pytest
from ase.atoms import Atoms
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from kumaranu.basisSetSelector import BasisSetSelector


project_root = Path(__file__).resolve().parents[2]


@pytest.fixture
def molecular_structure():
    return Atoms('CO2', positions=[[0.0, 0.0769, 0.0], [1.1797, -0.1620, -0.0], [-1.1797, 0.1043, 0.0]])


@pytest.fixture
def reference_datapoint():
    return Atoms('CO2', positions=[[0.0, 0.076925121247768, 0.0], [1.179724931716919, -0.162000626325607, -0.0], [-1.179724931716919, 0.104306787252426, 0.0]])


@pytest.fixture
def error_data_and_basis_sets() -> Tuple[Dict[str, List[float]], List[str]]:
    # Hardcoded data
    data = {
        "N2": [-1.2652515155542783, -0.4917908825196436, -0.03702286490532115, -0.08503021510053684,
               -0.08503021510053684, -0.06573130275116197, -0.11438153591093994, -0.11438153591093994,
               -0.11763637034154815, -0.12099438384939021],
        "CO": [-1.266834206595056, -0.475787564741829, -0.043394708420021647, -0.08409217074733753,
               -0.08409217074733753, -0.07535373422110433, -0.11529549705529316, -0.11529549705529316,
               -0.1179940151666584, -0.12031354017204007],
        "CO2": [-1.2997750766051088, -0.48029585658507695, -0.03818469310548968, -0.07979115363540501,
                -0.07979115363540501, -0.06835718570434102, -0.11187734731329128, -0.11187734731329128,
                -0.11498456847993427, -0.11686304503801831]
    }
    # this error-percent thing needs to be changed.
    basis_sets = [
        'STO-3G-error-percent', '3-21G-error-percent', '6-31G-error-percent',
        '6-31G*-error-percent', '6-31G**-error-percent', '6-311G-error-percent',
        '6-311G*-error-percent', '6-311G**-error-percent', '6-311++G**-error-percent',
        '6-311++G(2d,2p)-error-percent',
    ]

    return data, basis_sets


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
