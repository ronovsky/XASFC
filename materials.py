from dataclasses import dataclass, field, InitVar
from typing import Dict, List

import numpy as np

from data import ATOMIC_WEIGHTS, REFERENCE_SPECTRA


@dataclass
class Layer:

    name: str # Unique name for the layer
    depth: float  # in microns

    # Internal dict with densities of elements
    _densities: Dict[str, float] = field(init=False)

    # Initialization variables
    formula: InitVar[Dict[str, int] | None] = None
    density: InitVar[float | None] = None
    densities: InitVar[Dict[str, float] | None] = None

    def __post_init__(self, formula, density, densities):

        # Setup with a density or chemical formula
        if densities is not None:
            self._densities = densities
        else:
            assert formula is not None and density is not None

            total_atomic_weight = 0.0
            self._densities = {}

            # Compute total atomic weight per element
            for element, count in formula.items():
                self._densities[element] = count * ATOMIC_WEIGHTS[element]
                total_atomic_weight += self._densities[element]

            # Normalize to real density of the material
            for element, count in formula.items():
                self._densities[element] = self._densities[element] * density / total_atomic_weight

    def attn_coef(self, energies: np.ndarray) -> np.ndarray:

        # Compute the total attenuation coefficient of the layer
        total_attn = np.zeros_like(energies, dtype=float)
        for element, density in self._densities.items():

            if element not in REFERENCE_SPECTRA:
                print(f'Warning: {element} not in reference spectra')
            else:
                total_attn += self._densities[element] * REFERENCE_SPECTRA[element](energies)

        return total_attn


@dataclass
class Cell:

    layers: List[Layer]
    layer_depths: np.ndarray = field(init=False)
    total_depth: float = field(init=False)

    def __post_init__(self):
        self.layer_depths = np.array([layer.depth for layer in self.layers])
        self.total_depth = self.layer_depths.sum()

    def nickel_grid(self, grid_size: int, nickel_functions: Dict[str, callable]) -> np.ndarray:
        """
        Generate a nickel distribution grid for the entire cell based on layer-specific functions.

        Args:
            grid_size (int): Number of points in the grid.
            nickel_functions (Dict[str, callable]): A dictionary of layer names to functions defining
                                                    nickel distributions. Each function takes a 1D
                                                    array of normalized positions (0 to 1) within
                                                    the layer and returns the nickel values.

        Returns:
            np.ndarray: Nickel distribution over the entire cell.
        """
        nickel_grid = np.zeros(grid_size) # grid throughout the Cell
        layer_starts = np.cumsum(np.concatenate([[0], self.layer_depths[:-1]]))
        layer_ends = np.cumsum(self.layer_depths)
        # print('self.layers', self.layers)
        # print(layer_starts, layer_ends)

        for layer, start, end in zip(self.layers, layer_starts, layer_ends):
            layer_start_idx = int(start / self.total_depth * grid_size)
            layer_end_idx = int(end / self.total_depth * grid_size)
            layer_positions = np.linspace(0, 1, layer_end_idx - layer_start_idx)
            # print(layer_start_idx, layer_end_idx, layer_positions)
            # print('nickel_functions', nickel_functions)

            if layer.name in nickel_functions:
                # layer_nickel.size is a proportional size of the whole nickel_grid w/r Layer thickness
                # none of it is normalized, dunno if needed as I can normalize to the total amount
                # normalization to be checked
                layer_nickel = nickel_functions[layer.name](layer_positions)
                nickel_grid[layer_start_idx:layer_end_idx] = layer_nickel
                # print(layer.name)
                # print('layer_nickel', layer_nickel, layer_nickel.size, np.count_nonzero(layer_nickel))
                # print('nickel_grid', nickel_grid, nickel_grid.size, np.count_nonzero(nickel_grid))
            else:
                print(layer.name, ' NOT in nickel_functions')
        return nickel_grid

    def log_decay(self, energies: np.ndarray, depths: np.ndarray):

        attn_coefs = np.array([layer.attn_coef(energies) for layer in self.layers])
        bounds = np.cumsum(self.layer_depths) - self.layer_depths

        # The weights for a given depth are the distances spent in each layer to get there
        # axis 0 <-> desired_depths, axis 1 <-> weights
        weights = np.clip(depths[:, None] - bounds[None, :], 0.0, self.layer_depths[None, :])
        return weights @ attn_coefs


# Reference materials used in real experiments
CATHODE = Layer(name="CATHODE", depth=10.0, densities={'Pt': 0.130, 'Ni': 0.009, 'C': 0.509, 'F': 0.233, 'O': 0.027, 'S': 0.011})
CATHODE_no_Ni = Layer(name="CATHODE_no_Ni", depth=10.0, densities={'Pt': 0.130, 'C': 0.509, 'F': 0.233, 'O': 0.027, 'S': 0.011})
NAFION = Layer(name="NAFION", depth=25.0, formula={'H': 1, 'C': 9, 'O': 5, 'F': 17, 'S': 1}, density=1.96)
ANODE = Layer(name="ANODE", depth=4.0, densities={'Pt': 0.1975, 'C': 0.960, 'F': 0.421, 'O': 0.048, 'S': 0.019})
KAPTON = Layer(name="KAPTON", depth=65.0, formula={'H': 10, 'C': 22, 'O': 5, 'N': 2}, density=1.42)
GDL = Layer(name="GDL", depth=215.0, densities={'C': 0.326})


