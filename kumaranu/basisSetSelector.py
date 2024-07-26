import numpy as np
from ase.atoms import Atoms


class BasisSetSelector:
    def __init__(
            self,
            molecular_structure: Atoms,
            reference_datapoint: Atoms,
            tolerance: float,
            error_data: dict,
            basis_sets: list,
    ):
        self.molecular_structure = molecular_structure
        self.reference_datapoint = reference_datapoint
        self.tolerance = tolerance
        self.error_data = error_data
        self.basis_sets = basis_sets

    def select_basis_set(self):
        new_formula = str(self.molecular_structure.symbols)
        known_formula = str(self.reference_datapoint.symbols)

        if known_formula != new_formula:
            raise ValueError(
                f"The chemical formula for the new geometry ({new_formula}) and "
                f"the reference ({known_formula}) do not match.",
            )

        errors = np.abs(np.array(self.error_data[str(self.reference_datapoint.symbols)]))
        below_tolerance = errors <= self.tolerance

        if any(below_tolerance):
            selected_index = np.argmax(below_tolerance)
            selected_basis = self.basis_sets[selected_index][:-14]
            return selected_basis
        else:
            best_index = np.argmin(errors)
            best_basis = self.basis_sets[best_index][:-14]
            print(f"Warning: No basis set can satisfy the tolerance of {self.tolerance}. "
                  f"Using the best available basis set, {best_basis}.")
            return best_basis
