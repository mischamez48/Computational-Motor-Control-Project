
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
from plotting_common import plot_time_histories_multiple_windows

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise5_2():
    # -----Question 5.2----- #
    pylog.info("Ex 5")
    pylog.info("Implement exercise 5")
    log_path = './logs/exercise5/'
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
        feedback_weights_ipsi=0.25 * ws_ref,
        feedback_weights_contra=-0.25 * ws_ref,
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
    # 2.1 Phases
    plt.figure(figsize=(8,4))
    plt.plot(t, phases)
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Oscillator phases evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "phases.png"))
    plt.close()

    # 2.2 Amplitudes
    plt.figure(figsize=(8,4))
    plt.plot(t, amplitudes)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Oscilator amplitudes evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "amplitudes.png"))
    plt.close()

    # 2.3 Motor outputs and difference
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

    # 2.4 Joint angles evolution
    plt.figure(figsize=(8,4))
    for j in range(joint_angles.shape[1]):
        plt.plot(t, joint_angles[:, j], label=f"Articulation {j}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Joint angles evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "joint_angles.png"))
    plt.close()

    pylog.info(f"Plots saved in {log_path}")

def exercise5_3():

    pylog.info("Ex 5.3 - Compute metrics and record video")
    log_path = './logs/exercise5/'
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
        n_iterations=10000,
        video_record=True,
        video_name="exercise5_3",
        feedback_weights_ipsi=0.25 * ws_ref,
        feedback_weights_contra=-0.25 * ws_ref,
    )

    # Run the simulation
    controller = run_single(all_pars)

    # -----Extracting the data from the controller----- #
    metrics = controller.metrics
    pylog.info("=== Controller & Mechanical Metrics ===")
    for key, val in metrics.items():
        pylog.info(f"{key}: {val}")

    # -----Plots----- #
    joint_angles = controller.joints_positions[:, :all_pars.n_joints]
    t = np.arange(joint_angles.shape[0]) * all_pars.timestep

    n = all_pars.n_joints
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True)
    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = axs[r, c]
        if idx < n:
            ax.plot(t, joint_angles[:, idx], linewidth=1)
            ax.set_title(f"Joint {idx}", fontsize=10)
        else:
            ax.axis('off')
        if r == nrows - 1:
            ax.set_xlabel("Time (s)")
        if c == 0:
            ax.set_ylabel("Angle (rad)")
    fig.suptitle("Evolution of the joint angles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(log_path, "joint_angles.png"))
    plt.close(fig)

    pylog.info(f"Movie saved to {os.path.join(log_path, 'exercise5_swim.mp4')}")
    pylog.info(f"Plots and metrics saved in {log_path}")



if __name__ == '__main__':
    exercise5_2()
    # exercise5_3()

