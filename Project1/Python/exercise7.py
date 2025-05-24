import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from simulation_parameters import SimulationParameters
from util.run_open_loop import run_single
from util.entraining_signals import define_entraining_signals
from plotting_common import plot_time_histories, plot_1d

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise7():

    entraining_signals = define_entraining_signals(n_iterations=5001, frequency=8, amplitude_degrees=45, plot_signals=False)
    log_path = './logs/exercise7/'  # path for logging the simulation data
    # # os.makedirs(log_path, exist_ok=True)
    # print(entrainement_signal)
    # print(np.linspace(-1, 1, 20))

    all_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        log_path=log_path,
        print_metrics=False,
        return_network=True,
        headless=False,
        cpg_amplitude_gain = REF_JOINT_AMP[:-2], #[:-2]
        feedback_weights_ipsi = 0.0,
        feedback_weights_contra = 0.0,
        entraining_signals=entraining_signals,
    )

    controller = run_single(
        all_pars
    )

    neural_metrics = controller.metrics["neur_frequency"]
    print(neural_metrics)
    print("Neural frequency shape: ", neural_metrics.shape)
    # plot_1d(neural_metrics, [weight_name, "Neural Frequency [Hz]"])

if __name__ == '__main__':

    exercise7()
