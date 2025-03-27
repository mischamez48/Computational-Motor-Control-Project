
from plotting_common import plot_2d, plot_1d, save_figures, plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
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
        simulation_i=0,
        n_iterations=5001,
        controller="sine",
        amp=1.3,
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
    
    times = controller.times
    state = controller.motor_out
    left_idx = controller.motor_l
    right_idx = controller.motor_r

    plt.figure()
    plot_left_right(times, state, left_idx, right_idx, cm="jet", offset=0.3)
    plt.title("Left and Right Muscle Activations (Ml, Mr)")
    # plt.tight_layout()

    plt.figure("Head Trajectory")
    plot_trajectory(controller)
    # plt.tight_layout()

    
    joint_angles = controller.joints_positions
    n_joints = joint_angles.shape[1]
    joint_labels = [f"Joint {i}" for i in range(n_joints)]
    plot_time_histories_multiple_windows(times, joint_angles, labels=joint_labels, title="Joint Angles", closefig=False)
    # plt.tight_layout()

    plt.show()
    # controller = load_object("logs/example_single/controller0")

if __name__ == '__main__':
    exercise0()
    plt.show()
