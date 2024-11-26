import matplotlib.pyplot as plt
import numpy as np

from experiment_setup import CELL_Q, CELL_K, CELL_L, CELL_N, CELL_P, CELL_J, PELLETS_SECOND_BATCH, CELL_R, \
    PELLETS_FIRST_BATCH
from geometry import get_geometric_factors, get_cell_angle_distribution
from modelling import fit_experiments


def plot_single_experiment(experiment, coefs, params0, ax, geometric_correction, normalized_params):

    name = experiment.output_name or experiment.name  # (Possibly) a shorter name
    detectors = experiment.detectors
    reference_names = [name for name in coefs[detectors[0]] if name != 'background']

    # Compute geometric correction
    if geometric_correction is not None:
        factors = get_geometric_factors()
        # correction = factors['theoretical'] / factors[geometric_correction]
        print('model factors', factors['cell_R_model'])
        correction = factors['cell_R_model'] / params0
        print('correction', correction)
    print('params0', params0)

    params_Ni0 = np.array([coefs[detector]['PtNi-dealloyed'] for detector in detectors])
    factors = get_geometric_factors()
    correction_Ni0 = factors['cell_R_model'] / params_Ni0


    coefs_run = {}  # Rearranged coefficients by reference and detector
    for j, reference in enumerate(reference_names):

        params = np.array([coefs[detector][reference] for detector in detectors])
        print('params',params)

        # Normalize to Ni0 params for theoretical layer
        # params *= correction_Ni0

        # Normalize to 1
        if normalized_params is not None:
            params = params / params[0]

        if geometric_correction is not None:
            params *= correction  # noqa
            print('params_corr', params)

        ax[0].scatter(detectors, params, label=reference, color='rgb'[j % 3])
        coefs_run[reference] = params

    ax[0].legend()
    ax[0].set_title(name)

    ratio = coefs_run[reference_names[0]] / coefs_run[reference_names[1]]
    ax[1].scatter(detectors, ratio)


def plot_experiments(experiments, coefficients, geometric_correction=None, normalized_params=None):
    exp0 = experiments[0]
    coefs0 = coefficients[f'{exp0.name}{exp0.output_name}']
    name = exp0.output_name or exp0.name  # (Possibly) a shorter name
    detectors = exp0.detectors
    reference_names = [name for name in coefs0[detectors[0]] if name != 'background']
    params0 = np.array([coefs0[detector][reference_names[1]] for detector in detectors])
    # print('coefs0',coefs0)
    # print('params0', params0)

    # Setup plots
    fig, ax = plt.subplots(2, len(experiments))

    for i, experiment in enumerate(experiments):
        coefs = coefficients[f'{experiment.name}{experiment.output_name}']
        # print('coefs',coefs)
        plot_single_experiment(experiment, coefs, params0, [ax[0][i], ax[1][i]], geometric_correction, normalized_params)

    # Compute common axes
    coef_y_limit = max(ax_.get_ylim()[1] for ax_ in ax[0])
    ratio_y_limits = min(ax_.get_ylim()[0] for ax_ in ax[1]), max(ax_.get_ylim()[1] for ax_ in ax[1])

    for i in range(len(experiments)):
        ax[0][i].set_ylim([0.0, coef_y_limit])
        ax[1][i].set_ylim(ratio_y_limits)

    plt.show()


# def plot_experiments_with_norm_to_nth_cell(experiments, coefficients, n = 0, geometric_correction=None):
#     # Setup plots
#     fig, ax = plt.subplots(2, len(experiments))
#     coefs_norm_cell = coefficients[f'{experiments[n].name}{experiments[n].output_name}']
#     print(coefs_norm_cell)
#     for i, experiment in enumerate(experiments):
#         coefs = coefficients[f'{experiment.name}{experiment.output_name}']
#         print(coefs)
#         normalized_coefs = coefs / coefs_norm_cell
#         plot_single_experiment(experiment, normalized_coefs, [ax[0][i], ax[1][i]], geometric_correction)
#
#     # Compute common axes
#     coef_y_limit = max(ax_.get_ylim()[1] for ax_ in ax[0])
#     ratio_y_limits = min(ax_.get_ylim()[0] for ax_ in ax[1]), max(ax_.get_ylim()[1] for ax_ in ax[1])
#
#     for i in range(len(experiments)):
#         ax[0][i].set_ylim([0.0, coef_y_limit])
#         ax[1][i].set_ylim(ratio_y_limits)
#
#     plt.show()



if __name__ == '__main__':
    # cells_ = CELL_R[26:]
    cells_ = CELL_R[:3] + CELL_R[9:]
    # cells_ = CELL_R[:]
    # fit0 = fit_experiments([cells_[0]])
    # print(fit0)
    fit = fit_experiments(cells_, 0)
    print('fit',fit)
    plot_experiments(cells_, fit)#, 'cell_R_model') # plot_experiments(experiments, coefficients, geometric_correction=None, normalized_params=None)
    # plot_experiments_with_norm_to_cell(cells_, fit, )



'''
CELL_J:
    plotting error so far
CELL_K: 
    Ni0 stable, Ni2+ slight increase at first hold but then only decreasing for OCP and hold
    fit for 5th crystal not ideal as spectrum shifted to higher energies
CELL_L:
    first 2 rounds of OCP/hold do not show a clear trend but then OCP hold hold OCP is clear, Ni0 ca stable
CELL_N:
    not a clear trend, some holds high, some OCP high...
CELL_P:
    first hold: vastly different Ni2+/Ni0 ratio = 12 on C1-C3, 24 on C4 and 52 on C5?
    OCP: ratio < 1, hold ca 1, OCP << 1; this one might work.. Ni0 values are the same as 1st hold!!
CELL_Q:
    highly metallic, ratios << 1, Ni2+ content dropping, no clear trend, only 1st hold ratio = 1,
    Ni0 stable

'''