
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
from util.rw import load_object
import numpy as np

REF_JOINT_AMP = np.array([
    0.06580,
    0.02810,
    0.02781,
    0.03047,
    0.03623,
    0.04127,
    0.04864,
    0.05398,
    0.06508,
    0.08945,
    0.10271,
    0.11789,
    0.14929,
    0.0,      # Note: Tail moves passively,
    0.0,      # Note: Tail moves passively,
])  # type: ignore unit:radian


def exercise4():

    pylog.info("Implement ex 4")
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)


def simulate(trial, cpg_amplitude_gain, log_path):
    '''
    Here is a template function which you can use for single trial of simulation

    trial: <int>
        number of optimization iteration
    mo_gains_axial_old: <np.array> of shape n_active_joints
        nominal amplitude gain of the network that you have to optimize so that
        the resulting joint kinematics match the reference data
    '''

    all_pars = SimulationParameters(
        log_path=log_path,
        simulation_i=trial,
        headless=True,
        compute_metrics="all",
        print_metrics=False,
        controller="abstract oscillator",  # abstract oscillator
        n_iterations=5001,
        drive=10,
        cpg_amplitude_gain=cpg_amplitude_gain,
    )

    _ = run_single(
        all_pars
    )

# Hint: to load a controller from simulation result, use:
# controller = load_object('{}controller{}'.format(log_path, trial))


if __name__ == '__main__':
    exercise4()

