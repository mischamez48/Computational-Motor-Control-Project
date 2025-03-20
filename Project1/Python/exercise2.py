
import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
from util.run_closed_loop import run_multiple
from util.rw import load_object
from simulation_parameters import SimulationParameters
from plotting_common import plot_1d, plot_2d, save_figures


def exercise2():

    pylog.info("Implement ex 2")
    log_path = './logs/exercise2/'
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':

    exercise2()

