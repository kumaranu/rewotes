from typing import List
from ase.atoms import Atoms
from kumaranu.dataCollector import DataCollector
from kumaranu.basisSetSelector import BasisSetSelector


class BasisSetProvider:
    def __init__(
            self,
            project_root: str,
            tolerance: float,
            error_data_file: str = None,
            files_dir: str = None,
            basis_sets: List[str] = None,
            recalculate_errors: bool = False,
    ):
        self.project_root = project_root
        self.tolerance = tolerance
        self.error_data_file = error_data_file if error_data_file \
            else f'{self.project_root}/kumaranu/tests/molecule_xyz_files/basis_set_error_data.csv'
        self.files_dir = files_dir if files_dir else f'{self.project_root}/kumaranu/tests/molecule_xyz_files'
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
        if self.recalculate_errors:
            data_collector = DataCollector(
                self.project_root,
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
