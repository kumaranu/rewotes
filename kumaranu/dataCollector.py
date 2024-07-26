import os
import glob
import pandas as pd
from ase.io import read
from kumaranu.calculators import EnergyCalculator
from typing import List
from pathlib import Path


class DataCollector:
    """
    Collects and stores energy error data for various basis sets and molecular structures.

    Parameters
    ----------
    files_dir : str, optional
        The directory containing molecular XYZ files.
        If not provided, a default directory within the project root will be used.
    basis_sets : List[str], optional
        A list of basis sets to be considered.
        If not provided, a default list of common basis sets will be used.

    Methods
    -------
    collect_and_store_data()
        Collects energy error data for each molecular structure and basis set, and stores it in a CSV file.
    load_error_data(csv_file)
        Loads energy error data from a CSV file and returns it as a dictionary along with the basis sets.
    """
    def __init__(
            self,
            files_dir: str = None,
            basis_sets: List[str] = None,
    ):
        self.files_dir = files_dir if files_dir else f'{Path(__file__).resolve().parents[2]}/kumaranu/tests/molecule_xyz_files'
        self.basis_sets = basis_sets if basis_sets else [
            "STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**",
            "6-311G", "6-311G*", "6-311G**", "6-311++G**", "6-311++G(2d,2p)",
        ]

    def collect_and_store_data(self):
        """
        Collects energy error data for each molecular structure and basis set, and stores it in a CSV file.

        This method reads molecular structures from XYZ files, calculates the energy for each basis set,
        computes the error percentage, and saves the data to a CSV file in the specified directory.
        """
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
        """
        Loads energy error data from a CSV file.

        Parameters
        ----------
        csv_file : str
            The path to the CSV file containing the error data.

        Returns
        -------
        tuple
            A tuple containing:
                - error_data (dict): A dictionary with chemical names as keys and error percentages as values.
                - basis_sets (list of str): A list of basis sets.
        """
        df = pd.read_csv(csv_file)
        error_data = {}

        for _, row in df.iterrows():
            chemical_name = row['chemical_name']  # Convert string representation of list back to list
            errors = row[3:].values.astype(float)  # Get error values and convert to float
            error_data[chemical_name] = errors

        return error_data, df.columns[3:].tolist()  # return error data and basis sets
