import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELL_Q, CELL_K, CELL_L, CELL_N, CELL_P, CELL_J, PELLETS_SECOND_BATCH, CELL_R, \
    PELLETS_FIRST_BATCH
from geometry import get_geometric_factors
from modelling import fit_experiments


def plot_experiments(experiments, coefficients, geometric_correction=None):

    # Get the names of reference spectra
    first_coef = next(iter(coefficients.values()))
    reference_names = [name for name in first_coef if name != 'background']

    if geometric_correction is not None:
        factors = get_geometric_factors()
        correction = factors['theoretical'] / factors[geometric_correction]

    fig, ax = plt.subplots(2, len(experiments))
    colors = ['red', 'green', 'blue']

    coef_y_limit = 0.0
    ratio_y_limits = [np.inf, 0]

    for i, experiment in enumerate(experiments):

        name = experiment.output_name or experiment.name  # (Possibly) a shorter name
        detectors = experiment.detectors

        coefs_run = {}
        for j, reference in enumerate(reference_names):
            coefs_run[reference] = np.array([coefficients[(name, detector)][reference] for detector in detectors])

            if geometric_correction is not None:
                coefs_run[reference] *= correction

            ax[0][i].scatter(detectors, coefs_run[reference], label=reference, color=colors[j % 3])
            coef_y_limit = max(coef_y_limit, np.max(coefs_run[reference]))

        ax[0][i].legend()
        ax[0][i].set_title(name)

        ratio = coefs_run[reference_names[0]] / coefs_run[reference_names[1]]
        ax[1][i].scatter(detectors, ratio)
        ratio_y_limits[0] = min(ratio_y_limits[0], np.min(ratio))
        ratio_y_limits[1] = max(ratio_y_limits[1], np.max(ratio))

    for i in range(len(experiments)):
        ax[0][i].set_ylim([0.0, 3.0])
        ax[1][i].set_ylim(ratio_y_limits)

    plt.show()


if __name__ == '__main__':

    cells_ = CELL_R[:3] + CELL_R[9:]
    fit = fit_experiments(cells_, -1)

    plot_experiments(cells_, fit)

