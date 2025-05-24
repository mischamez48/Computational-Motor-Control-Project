from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
from plotting_common import plot_time_histories

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise9():

    pylog.info("Ex 9")
    pylog.info("Implement exercise 9")
    log_path = './logs/exercise9/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':

    exercise9()
