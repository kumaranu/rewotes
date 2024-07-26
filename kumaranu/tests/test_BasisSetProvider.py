import os
import pytest
import shutil
from ase.io import read
from pathlib import Path

from kumaranu.basisSetProvider import BasisSetProvider

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


def test_BasisSetProvider(
        setup_test_environment1,
        setup_test_environment2,
):
    reference = setup_test_environment1
    new_structure = setup_test_environment2

    basisSetProviderObject = BasisSetProvider(
        project_root,
        tolerance=0.1,
    )
    assert basisSetProviderObject.get_basis_set(
        new_structure,
        reference
    ) == '6-31G'


def test_BasisSetProvider_low_tol(
        setup_test_environment1,
        setup_test_environment2,
):
    reference = setup_test_environment1
    new_structure = setup_test_environment2

    basisSetProviderObject = BasisSetProvider(
        project_root,
        tolerance=0.01,
    )
    assert basisSetProviderObject.get_basis_set(
        new_structure,
        reference
    ) == '6-31G'


def test_BasisSetProvider_high_tol(
        setup_test_environment1,
        setup_test_environment2,
):
    reference = setup_test_environment1
    new_structure = setup_test_environment2

    basisSetProviderObject = BasisSetProvider(
        project_root,
        tolerance=5,
    )
    assert basisSetProviderObject.get_basis_set(
        new_structure,
        reference
    ) == 'STO-3G'
