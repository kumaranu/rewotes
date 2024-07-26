import glob
import numpy as np
import pandas as pd
from ase.io import read
from ase.atoms import Atoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.calculators.nwchem import NWChem
from ase.thermochemistry import IdealGasThermo


class ErrorDataLoader:
    @staticmethod
    def load_error_data(csv_file: str):
        df = pd.read_csv(csv_file)
        error_data = {}
        basis_sets = df.columns[3:].tolist()

        for _, row in df.iterrows():
            chemical_name = row['chemical_name']
            errors = row[3:].values.astype(float)  # Get error values and convert to float
            error_data[chemical_name] = errors

        return error_data, basis_sets
