
from plotting_common import plot_time_histories, save_figures
from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
import numpy as np
import matplotlib
matplotlib.rc('font', **{"size": 15})


num_process = 24  # number of processes to run in parallel
ylim_amp = [0, 0.01]


def exercise1():

    pylog.info("Ex 1")
    pylog.info("Implement exercise 1")
    prepath = './logs/exercise1/'


if __name__ == '__main__':

    exercise1()

