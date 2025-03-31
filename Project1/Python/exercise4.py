
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

def GradientDescent(lr = 0.1, itermax = 100, tolerance = 0.01):
    
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    cpg_amplitude_gain = 0.125 * np.ones(13)
    simulate(0, cpg_amplitude_gain, log_path)
    controller = load_object('{}controller{}'.format(log_path, 0))
    A_res = controller.metrics["mech_joint_amplitudes"]

    iter = 1
    error_joint_amp = np.linalg.norm(REF_JOINT_AMP-A_res)

    while error_joint_amp > tolerance and iter<itermax:
        cpg_amplitude_gain = cpg_amplitude_gain*(1 + lr*(-1+REF_JOINT_AMP[:-2]/A_res[:-2]))
        simulate(iter, cpg_amplitude_gain, log_path)
        controller = load_object('{}controller{}'.format(log_path, iter))
        A_res = controller.metrics["mech_joint_amplitudes"]
        error_joint_amp = np.linalg.norm(REF_JOINT_AMP-A_res)
        iter += 1

        if iter%10==0:
            print(f"Error at step {iter} : {error_joint_amp}")

    if iter<itermax:
        print(f"Number max of iteration reached. Error of {error_joint_amp}")

    if error_joint_amp > tolerance:
        print(f"Error below tolerance reached after {iter} iterations. Error of {error_joint_amp}")

    return cpg_amplitude_gain

def exercise4():

    pylog.info("Implement ex 4")
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    cpg_amplitude_gain = GradientDescent(lr = 0.15, itermax = 100, tolerance = 0.01)

    print(cpg_amplitude_gain)


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

