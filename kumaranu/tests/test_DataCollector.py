import pytest
import os
import pandas as pd
from ase import Atoms
from ase.io import write
from kumaranu.dataCollector import DataCollector
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup_test_environment(tmp_path):
    # Create a temporary directory for the test
    test_dir = tmp_path / "test_project"
    test_dir.mkdir()

    # Create a dummy XYZ file
    mol_file = test_dir / "N2_first.xyz"
    n2 = Atoms('N2', positions=[[0.0, 0.0, 0.559], [0.0, 0.0, -0.559]])
    n2.info = {"ref_energy": -109.482}  # Example reference energy
    write(mol_file, n2)

    files_dir = project_root / "kumaranu" / "tests" / 'three_molecules'

    basis_sets = [
        "STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**",
        "6-311G", "6-311G*", "6-311G**", "6-311++G**", "6-311++G(2d,2p)",
    ]

    return mol_file, str(files_dir), basis_sets


def test_collect_and_store_data(setup_test_environment):
    mol_file, files_dir, basis_sets = setup_test_environment

    # Create DataCollector instance
    collector = DataCollector(files_dir, basis_sets)

    collector.collect_and_store_data()

    output_file = files_dir + '/basis_set_error_data.csv'

    # Check if the CSV file is created
    assert os.path.exists(output_file)

    # Load the data and check if it matches expected values
    df = pd.read_csv(output_file)
    assert 'chemical_name' in df.columns
    assert 'chemical_symbols' in df.columns
    assert 'geometry' in df.columns
    assert 'STO-3G-error-percent' in df.columns

    row = df.iloc[0]
    assert row['chemical_name'] == 'N2'
    assert eval(row['chemical_symbols']) == ['N', 'N']
    assert eval(row['geometry'])[0][2] == pytest.approx(
        0.559859335422516,
        abs=1e-6,
    )
    assert row['STO-3G-error-percent'] is not None


def test_load_error_data(setup_test_environment):
    mol_file, files_dir, basis_sets = setup_test_environment

    # Create DataCollector instance
    collector = DataCollector(files_dir, basis_sets)

    # Create a mock CSV file
    data = {
        'chemical_name': ['N2'],
        'chemical_symbols': ["['N', 'N']"],
        'geometry': ["[[0.0, 0.0, 0.559], [0.0, 0.0, -0.559]]"],
        'STO-3G-error-percent': 0.5,
        '3-21G-error-percent': 0.4,
        '6-31G-error-percent': 0.3,
    }

    output_file = files_dir + '/basis_set_error_data.csv'

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    # Load the error data
    error_data, basis_sets = collector.load_error_data(str(output_file))

    # Check if the loaded data matches the expected values
    assert 'N2' in error_data
    assert len(error_data['N2']) == len(basis_sets)
    assert error_data['N2'][0] == 0.5  # STO-3G-error-percent
    assert error_data['N2'][1] == 0.4  # 3-21G-error-percent
    assert error_data['N2'][2] == 0.3  # 6-31G-error-percent
    assert basis_sets == [
        'STO-3G-error-percent', '3-21G-error-percent', '6-31G-error-percent'
    ]
