import pytest
from kumaranu.basis_set_provider import get_basis_set


def test_get_basis_set_high_precision():
    basis_set = get_basis_set("H2O", 0.0, 0.005)
    assert basis_set == "cc-pVTZ"


def test_get_basis_set_medium_precision():
    basis_set = get_basis_set("H2O", 0.0, 0.03)
    assert basis_set == "cc-pVDZ"


def test_get_basis_set_low_precision():
    basis_set = get_basis_set("H2O", 0.0, 0.1)
    assert basis_set == "STO-3G"
