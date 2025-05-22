
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


def exercise5():

    pylog.info("Ex 5")
    pylog.info("Implement exercise 5")
    log_path = './logs/exercise5/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    # Simulation parameters
    all_pars = SimulationParameters(
        log_path=log_path,
        simulation_i=0,
        headless=True,
        compute_metrics="all",
        print_metrics=False,
        return_network=True,
        controller="abstract oscillator",
        n_iterations=1000,
    )

    controller = run_single(all_pars)

    # -----Extracting the data from the controller----- #
    n_osc = controller.n_oscillators
    # Phases
    phases = controller.state[:, : n_osc]
    # Amplitudes
    amplitudes = controller.state[:, n_osc:2 * n_osc]
    # Motor outputs
    motor = controller.motor_out
    # Difference between left and right motor outputs
    motor_diff = motor[:, 0::2] - motor[:, 1::2]
    # Joint angles
    joint_angles = controller.joints_positions[:, : all_pars.n_joints]
    # Temporal axis
    t = np.arange(phases.shape[0]) * all_pars.timestep

    # -----Plotting the results----- #
    # 3.1 Phases
    plt.figure(figsize=(8,4))
    plt.plot(t, phases)
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Oscillator phases evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "phases.png"))
    plt.close()

    # 3.2 Amplitudes
    plt.figure(figsize=(8,4))
    plt.plot(t, amplitudes)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Oscilator amplitudes evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "amplitudes.png"))
    plt.close()

    # 3.3 Motor outputs and difference
    plt.figure(figsize=(8,4))
    for j in range(motor.shape[1]):
        plt.plot(t, motor[:, j], alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle activation")
    plt.title("Motor output (ML, MR)")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "motor_output.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(t, motor_diff)
    plt.xlabel("Time (s)")
    plt.ylabel("ML - MR")
    plt.title("Difference between left and right motor outputs")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "motor_diff.png"))
    plt.close()

    # 3.4 Joint angles evolution
    plt.figure(figsize=(8,4))
    for j in range(joint_angles.shape[1]):
        plt.plot(t, joint_angles[:, j], label=f"Articulation {j}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Joint angles evolution")
    plt.legend(ncol=3, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "joint_angles.png"))
    plt.close()

    pylog.info(f"Plots saved in {log_path}")

if __name__ == '__main__':

    exercise5()

