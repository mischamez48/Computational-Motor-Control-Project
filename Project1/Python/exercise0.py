
import plotting_common
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
import numpy as np
import matplotlib
matplotlib.rc('font', **{"size": 15})


def exercise0():

    pylog.info("Implement ex 0")
    log_path = './logs/exercise0/'
    os.makedirs(log_path, exist_ok=True)

    pars = SimulationParameters(
        n_iterations=5001,
        controller="sine",
        amp=0.3,
        twl=1,
        freq=3,
        compute_metrics="all",
        headless=True,
        video_record=False,
        log_path=log_path,
        return_network=True,
    )

    controller = run_single(
        pars
    )


if __name__ == '__main__':

    exercise0()

