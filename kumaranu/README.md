# Basis set selector (Chemistry)

> Ideal candidate: scientists skilled in Density Functional Theory and proficient in python.

# Overview

The aim of this task is to create a simple python package that implements automatic basis set selection mechanism for a quantum chemistry engine.

# Requirements

1. automatically find the basis set delivering a particular precision, passed as argument (eg. within 0.01% from reference)
1. use either experimental data or higher-fidelity modeling results (eg. coupled cluster) as reference data
1. example properties to converge: HOMO-LUMO gaps, vibrational frequencies

# Expectations

- mine reference data for use during the project
- correctly find a basis set that satisfies a desired tolerance for a set of 10-100 molecules, starting from H2, as simplest, up to a 10-20-atom ones
- modular and object-oriented implementation
- commit early and often - at least once per 24 hours

# Timeline

We leave exact timing to the candidate. Must fit Within 5 days total.

# User story

As a user of this software I can start it passing:

- molecular structure
- reference datapoint
- tolerance (precision)

as parameters and get the basis set that satisfies the tolerance criterion.

# Notes

- create an account at exabyte.io and use it for the calculation purposes
- suggested modeling engine: NWCHEM or SIESTA

## Getting Started

### Clone the Repository:

```
git clone https://github.com/kumaranu/rewotes.git

cd rewotes
```
Create and Activate Conda Environment:
```
conda create -n test0 python=3.10
conda activate test0
```
Install the Package:
```
pip install -e .
```
Run the python script given below:
```
import importlib
from pathlib import Path
from ase.atoms import Atoms
from kumaranu.basisSetProvider import BasisSetProvider

# Define project_root
kumaranu_spec = importlib.util.find_spec('kumaranu')
project_root = Path(kumaranu_spec.origin).parent.parent

# Define the molecules
mol = Atoms('CO2', positions=[[0, 0, 0], [1, 1.01, 1], [-1, -1.03, -1]])
ref = Atoms('CO2', positions=[[0, 0, 0], [1.01, 1, 1], [-1, -1.01, -1]])

# Set the tolerance
tolerance = 0.5

# Create the BasisSetProvider object
basisProviderObject = BasisSetProvider(
    tolerance,
    files_dir=str(project_root / 'kumaranu/tests/three_molecules'),
    recalculate_errors=True,
)

# Get the selected basis set
selected_basis = basisProviderObject.get_basis_set(mol, ref)
print(selected_basis)
```
