from ase.atoms import Atoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.calculators.nwchem import NWChem
from ase.thermochemistry import IdealGasThermo


class EnergyCalculator:
    """
    Calculates the energy of a molecular structure using the NWChem calculator.

    Methods
    -------
    calculate_energy(ase_obj, basis, vib_analysis=False, **kwargs)
        Calculates the potential energy or Gibbs free energy of a given molecular structure.
    """
    @staticmethod
    def calculate_energy(
            ase_obj: Atoms,
            basis: str,
            vib_analysis=False,
            **kwargs,
    ):
        """
        Calculates the potential energy or Gibbs free energy of a given molecular structure.

        Parameters
        ----------
        ase_obj : Atoms
            The ASE Atoms object representing the molecular structure.
        basis : str
            The basis set to be used for the energy calculation.
        vib_analysis : bool, optional
            If True, perform vibrational analysis and calculate Gibbs free energy (default is False).
        **kwargs : dict, optional
            Additional parameters for the vibrational and thermochemical analysis, including:
            - geometry : str
                The molecular geometry for thermochemical analysis ('linear' or 'nonlinear', default is 'nonlinear').
            - symmetrynumber : int
                The symmetry number for the molecule (default is 1).
            - spin : int
                The spin multiplicity of the molecule (default is 0).
            - temperature : float
                The temperature in Kelvin for Gibbs free energy calculation (default is 298.15).
            - pressure : float
                The pressure in Pascals for Gibbs free energy calculation (default is 101325).

        Returns
        -------
        float
            The calculated potential energy or Gibbs free energy of the molecule.

        Notes
        -----
        - If `vib_analysis` is True, the molecule's geometry will be optimized before performing vibrational analysis.
        - The potential energy is calculated using Density Functional Theory (DFT) with the B3LYP exchange-correlation functional.
        """
        ase_obj.calc = NWChem(
            dft=dict(
                maxiter=2000,
                xc='B3LYP',
                convergence={
                    'energy': 1e-9,
                    'density': 1e-7,
                    'gradient': 5e-6,
                    'hl_tol': 0.01,
                }
            ),
            basis=basis,
        )
        if vib_analysis:
            dyn = QuasiNewton(ase_obj)
            dyn.run(fmax=0.01)
            energy = ase_obj.get_potential_energy()
            vib = Vibrations(ase_obj)
            vib.run()
            vib_energies = vib.get_energies()
            thermo = IdealGasThermo(
                vib_energies=vib_energies,
                geometry=kwargs.get('geometry', 'nonlinear'),
                potentialenergy=energy,
                atoms=ase_obj,
                symmetrynumber=kwargs.get('symmetrynumber', 1),
                spin=kwargs.get('spin', 0),
            )
            G = thermo.get_gibbs_energy(
                temperature=kwargs.get('temperature', 298.15),
                pressure=kwargs.get('pressure', 101325.),
            )
            return G
        return ase_obj.get_potential_energy()

# Can add more functionalities here in the future.
