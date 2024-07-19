class BasisSetProvider:
    def __init__(self, molecular_structure, reference_datapoint, tolerance):
        self.molecular_structure = molecular_structure
        self.reference_datapoint = reference_datapoint
        self.tolerance = tolerance

    def select_basis_set(self):
        # Starting without electronic structure calculations
        # and simply assigning basis depending on the tolerance value
        # Later on will add electronic strucuture calcuations and comparison with the reference data.
        if self.tolerance < 0.01:
            return "cc-pVTZ"
        elif self.tolerance < 0.05:
            return "cc-pVDZ"
        else:
            return "STO-3G"


def get_basis_set(molecular_structure, reference_datapoint, tolerance):
    provider = BasisSetProvider(molecular_structure, reference_datapoint, tolerance)
    return provider.select_basis_set()
