from ase.atoms import Atoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.calculators.nwchem import NWChem
from ase.thermochemistry import IdealGasThermo


class EnergyCalculator:
    @staticmethod
    def calculate_energy(ase_obj: Atoms, basis: str, vib_analysis=False, **kwargs):
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
