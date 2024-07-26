from typing import List
from ase.atoms import Atoms
from kumaranu.dataCollector import DataCollector
from kumaranu.basisSetSelector import BasisSetSelector
from pathlib import Path


class BasisSetProvider:
    """
    Provides the best basis set for a given molecular structure within a specified tolerance.

    Parameters
    ----------
    tolerance : float
        The tolerance for selecting the basis set.
    error_data_file : str, optional
        The path to the CSV file containing error data for basis sets.
        If not provided, a default path within the project root will be used.
    files_dir : str, optional
        The directory containing molecular XYZ files.
        If not provided, a default directory within the project root will be used.
    basis_sets : List[str], optional
        A list of basis sets to be considered.
        If not provided, a default list of common basis sets will be used.
    recalculate_errors : bool, optional
        Whether to recalculate errors and update the error data file.
        Default is False.

    Methods
    -------
    get_basis_set(molecular_structure, reference_datapoint)
        Returns the best basis set for the given molecular structure within the specified tolerance.
    """
    def __init__(
            self,
            tolerance: float,
            error_data_file: str = None,
            files_dir: str = None,
            basis_sets: List[str] = None,
            recalculate_errors: bool = False,
    ):
        self.tolerance = tolerance
        self.files_dir = files_dir if files_dir else f'{Path(__file__).resolve().parents[2]}/kumaranu/tests/molecule_xyz_files'
        self.error_data_file = error_data_file if error_data_file \
            else f'{files_dir}/basis_set_error_data.csv'
        self.basis_sets = basis_sets if basis_sets else [
            "STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**",
            "6-311G", "6-311G*", "6-311G**", "6-311++G**", "6-311++G(2d,2p)",
        ]
        self.recalculate_errors = recalculate_errors

    def get_basis_set(
            self,
            molecular_structure: Atoms,
            reference_datapoint: Atoms,
    ) -> str:
        """
        Returns the best basis set for the given molecular structure within the specified tolerance.

        Parameters
        ----------
        molecular_structure : Atoms
            The molecular structure for which to select the basis set.
        reference_datapoint : Atoms
            The reference molecular structure to compare against.

        Returns
        -------
        str
            The selected basis set within the specified tolerance.
        """
        if self.recalculate_errors:
            data_collector = DataCollector(
                self.files_dir,
                self.basis_sets,
            )
            data_collector.collect_and_store_data()
        error_data, basis_sets = DataCollector.load_error_data(self.error_data_file)

        selector = BasisSetSelector(
            molecular_structure,
            reference_datapoint,
            self.tolerance,
            error_data,
            basis_sets,
        )
        return selector.select_basis_set()
