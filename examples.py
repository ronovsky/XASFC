from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import PELLETS_SECOND_BATCH, CELL_J, CELL_I
from materials import Layer, Cell, ANODE, NAFION, CATHODE, CATHODE_no_Ni, GDL, KAPTON
from data import load_experiment_data
from intensity import output_intensity


def run_angle_comparison_naf212_ni_au():

    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    gold_density = 19.3
    gold = Layer(depth=14.0 / gold_density * 1e-2,  # conversion cm -> m
                 densities={'Au': gold_density})

    layers = [
        ANODE,
        gold,
        NAFION,
        gold,
        NAFION,
        gold,
        ANODE,
    ]

    cell = Cell(layers)

    energies = np.array([8400])
    energy_out = 7480

    grid_size = 10000

    # Setup hypothetical densities (for angle distribution, only the relative placement matters)
    nickel = {'top': np.zeros(grid_size),
              'bottom': np.zeros(grid_size),
              'original': np.zeros(grid_size)}

    nickel['top'][:int(0.05 * grid_size)] = 1.0
    nickel['bottom'][-int(0.05 * grid_size):] = 1.0

    nickel['uniform'] = np.ones(grid_size)

    # Originally, the nickel is located near the gold
    nickel['original'][int(4 / 108 * grid_size)] = 12.0
    nickel['original'][int(54 / 108 * grid_size)] = 12.0
    nickel['original'][int(104 / 108 * grid_size)] = 11.0

    results = {name: output_intensity(theta_in, theta_out, cell, energies, energy_out, density)
               for name, density in nickel.items()}

    # real data
    cell = CELL_I[0]
    data = load_experiment_data(cell)

    # Band of energies close to 8400 ev
    results[f'real_data: {cell.output_name or cell.name}'] = data.values[777-10:777+10, :5].mean(axis=0).reshape((1, -1))

    fig, ax = plt.subplots()
    for name, result in results.items():
        ax.plot(cell.detectors, result[0, :] / result[0, 0], label=name)

    plt.legend()
    plt.show()


def run_ls_04_ccm_comparison(norm=['amount','first1']):
    platinum_density = 21.447

    cell = Cell(layers=[
        # KAPTON,
        # GDL,
        CATHODE,
        # Layer(depth=150.0 / platinum_density * 1e-2, densities={'Pt': platinum_density}),
        NAFION, # Layer(depth=25.0, formula={'H': 1, 'C': 9, 'O': 5, 'F': 17, 'S': 1}, density=1.96)
        # Layer(depth=4.0, formula={'H': 1, 'C': 9, 'O': 5, 'F': 17, 'S': 1}, density=1.96)
        ANODE, # Layer(depth=4.0, densities={'Pt': 0.1975, 'C': 0.960, 'F': 0.421, 'O': 0.048, 'S': 0.019})
    ])

    # Define example nickel distribution functions ... can be moved to separate file
    def uniform_distribution(x):
        return np.ones_like(x)

    def top_distribution(x,percent=0.05):
        nickel = np.zeros_like(x)
        nickel[x < percent] = 1.0
        return nickel

    def gradient_distribution(x):
        return x  # Linearly increasing nickel concentration

    def zeros(x):
        return np.zeros_like(x)

    grid_size = 10000
    nickel_distributions = { # name : nickel functions
        'test1':{
            'CATHODE': gradient_distribution,
            'NAFION': top_distribution,
            'ANODE': top_distribution
        },
        'test2':{
            'CATHODE': uniform_distribution,
            'NAFION': top_distribution,
            'ANODE': top_distribution
        },
        'cathode-ones,else-zeros': {
            'CATHODE': partial(top_distribution, percent=CATHODE.depth/100*(cell.total_depth*0.05)),# or lambda x: top_distribution(x, portion=0.1)
            # 'NAFION': zeros,
            # 'ANODE': zeros
        },
    }

    nickel = {name: cell.nickel_grid(grid_size, nickel_distribution) \
              for name, nickel_distribution in nickel_distributions.items()}

    # old way to define nickel distribution
    nickel['top'] = np.zeros(grid_size)
    nickel['bottom'] = np.zeros(grid_size)
    nickel['middle'] = np.zeros(grid_size)

    nickel['top'][:int(0.05 * grid_size)] = 1.0
    nickel['bottom'][-int(0.05 * grid_size):] = 1.0
    middle_start = int((grid_size - 0.05 * grid_size) / 2)
    middle_end = middle_start + int(0.05 * grid_size)
    nickel['middle'][middle_start:middle_end] = 1.0

    nickel['uniform'] = np.ones(grid_size)
    # print(nickel['original'])
    # print(nickel['top'])

    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    energies = np.array([8380])
    energy_out = 7480

    if 'amount' in norm:
        # Compute normalization for each density array individually
        normalization_dens = {name: np.sum(density) for name, density in nickel.items()}
    else:
        normalization_dens = {name: 1 for name in nickel.keys()}  # No normalization

    results = {name: output_intensity(theta_in, theta_out, cell, energies, energy_out,\
                                      density / normalization_dens[name]) # /sum(density)
               for name, density in nickel.items()}

    in_cell = load_experiment_data(CELL_J[0])
    ex_situ = load_experiment_data(PELLETS_SECOND_BATCH[9])

    # results['in_cell'] = in_cell.values[575 - 10:575 + 10, :5].mean(axis=0).reshape((1, -1))
    # results['ex_situ'] = ex_situ.values[585 - 10:585 + 10, :5].mean(axis=0).reshape((1, -1))

    if 'first' in norm:
        normalization_crystal = {name: result[0, 0] for name, result in results.items()}
    else:
        normalization_crystal = {name: 1 for name in results.keys()}

    fig, ax = plt.subplots()
    for name, result in results.items():
        resulted_coefs = result[0, :] / normalization_crystal[name]
        print(resulted_coefs)
        ax.plot(CELL_J[0].detectors, resulted_coefs, label=name) #

    plt.legend()
    plt.show()


def run_metalic_Ni_layer_in_cathode():
    platinum_density = 21.447

    cell = Cell(layers=[
        #Layer(depth=150.0 / platinum_density * 1e-2, densities={'Pt': platinum_density}),
        GDL,
        CATHODE_no_Ni,
        #CATHODE_no_Ni,
        #CATHODE_no_Ni,
        #CATHODE_no_Ni
        #NAFION,
        #ANODE,
        #GDL
    ])
    print(cell.total_depth)
    theta_in = np.pi / 180 * 31
    theta_out = np.pi / 180 * np.array([41, 34, 27, 20, 13])  # Detectors C1-5

    energies = np.array([8380])
    energy_out = 7480

    grid_size = 10000
    cathode_part = grid_size*(1-CATHODE_no_Ni.depth/cell.total_depth)
    print(cathode_part)
    nickel = {#'top': np.zeros(grid_size),
              #'bottom': np.zeros(grid_size),
              'original': np.zeros(grid_size)}

    #nickel['top'][:int(0.05 * grid_size)] = 1.0
    #nickel['bottom'][-int(0.05 * grid_size):] = 1.0
    #nickel['original'][:int(1 * grid_size)] = 1.0 # only
    nickel['original'][int(cathode_part):] = 1
    #print(nickel['original'])
    #print(nickel['top'])
    results = {name: output_intensity(theta_in, theta_out, cell, energies, energy_out, density)
               for name, density in nickel.items()}
    print(results)
    #in_cell = load_experiment_data(CELL_J[0])
    #ex_situ = load_experiment_data(PELLETS_SECOND_BATCH[9])

    #results['in_cell'] = in_cell.values[575 - 10:575 + 10, :5].mean(axis=0).reshape((1, -1))
    #results['ex_situ'] = ex_situ.values[585 - 10:585 + 10, :5].mean(axis=0).reshape((1, -1))

    fig, ax = plt.subplots()
    for name, result in results.items():
        ax.plot(CELL_J[0].detectors, result[0, :] / result[0, 0], label=name)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # run_metalic_Ni_layer_in_cathode()
    run_ls_04_ccm_comparison()
    # run_angle_comparison_naf212_ni_au()