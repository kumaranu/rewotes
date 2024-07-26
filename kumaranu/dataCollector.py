import os
import glob
import pandas as pd
from ase.io import read
from kumaranu.calculators import EnergyCalculator
from typing import List


class DataCollector:
    def __init__(
            self,
            project_root: str,
            files_dir: str = None,
            basis_sets: List[str] = None,
    ):
        self.project_root = project_root
        self.files_dir = files_dir if files_dir else f'{self.project_root}/kumaranu/tests/molecule_xyz_files'
        self.basis_sets = basis_sets if basis_sets else [
            "STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**",
            "6-311G", "6-311G*", "6-311G**", "6-311++G**", "6-311++G(2d,2p)",
        ]

    def collect_and_store_data(self):
        data = []
        mol_list = glob.glob(f'{self.files_dir}/*_first.xyz')

        for mol in mol_list:
            input_ase_obj = read(mol)
            ref_energy = float(list(input_ase_obj.info)[1])
            row_data = {
                'chemical_name': input_ase_obj.symbols,
                'chemical_symbols': input_ase_obj.get_chemical_symbols(),
                'geometry': input_ase_obj.positions.tolist(),
            }

            for basis in self.basis_sets:
                try:
                    energy = EnergyCalculator.calculate_energy(input_ase_obj, basis)
                    err_percent = (abs(energy / 27.2114 - ref_energy) / ref_energy) * 100
                    row_data[basis + '-error-percent'] = err_percent
                except Exception as e:
                    print(f"An error occurred with basis set {basis}: {e}")
                    row_data[basis + '-error-percent'] = None
            data.append(row_data)
            print(f'Done with {mol}.')

        os.makedirs(self.files_dir, exist_ok=True)

        df = pd.DataFrame(data)
        df.to_csv(f'{self.files_dir}/basis_set_error_data.csv', index=False)
        print(f'Data has been saved to {self.files_dir}/basis_set_error_data.csv')

    @staticmethod
    def load_error_data(csv_file):
        df = pd.read_csv(csv_file)
        error_data = {}

        for _, row in df.iterrows():
            chemical_name = row['chemical_name']  # Convert string representation of list back to list
            errors = row[3:].values.astype(float)  # Get error values and convert to float
            error_data[chemical_name] = errors

        return error_data, df.columns[3:].tolist()  # return error data and basis sets
