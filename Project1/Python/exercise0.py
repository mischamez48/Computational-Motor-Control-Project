
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

    # Parameter ranges
    epsilon_values = np.linspace(0, 2, 3)  # Example: [0, 1, 2]
    amplitude_values = np.linspace(0, 2, 3)  # Example: [0, 1, 2]
    frequency_values = np.linspace(1, 5, 3)  # Example: [1, 3, 5] Hz

    for epsilon in epsilon_values:
        for A in amplitude_values:
            for f in frequency_values:
                pylog.info(f"Running simulation for ε={epsilon}, A={A}, f={f}")

                pars = SimulationParameters(
                    n_iterations=5001,
                    controller="sine",
                    amp=A,
                    twl=epsilon,
                    freq=f,
                    compute_metrics="none",
                    headless=True,
                    video_record=False,
                    log_path=log_path,
                    return_network=True,
                )
                
                controller = run_single(pars)
                
                # Extract data
                times = np.array(controller.times)
                joint_angles = np.array(controller.joint_angles)
                head_positions = np.array(controller.links_positions)[:, 0, :2]
                muscle_activations = np.array(controller.motor_outputs)
                left_activations = muscle_activations[:, :15]
                right_activations = muscle_activations[:, 15:]

                # Plot Muscle Activations
                plt.figure(figsize=(12, 8))
                plt.suptitle(f"Muscle Activations (ε={epsilon}, f={f}, A={A})")
                plotting_common.plot_left_right(times, muscle_activations, left_idx=range(15), right_idx=range(15, 30))
                plt.tight_layout()
                plt.savefig(f"{log_path}/muscle_activations_e{epsilon}_A{A}_f{f}.png")

                # Plot Joint Angles Evolution
                plt.figure(figsize=(12, 6))
                plt.title(f"Joint Angles Evolution (ε={epsilon}, f={f}, A={A})")
                plotting_common.plot_time_histories(times, joint_angles)
                plt.tight_layout()
                plt.savefig(f"{log_path}/joint_angles_e{epsilon}_A{A}_f{f}.png")

                # Plot Head Trajectory
                plt.figure(figsize=(10, 8))
                plt.title(f"Head Trajectory (ε={epsilon}, f={f}, A={A})")
                plotting_common.plot_trajectory(controller, label="Head trajectory", color='blue')
                plt.tight_layout()
                plt.savefig(f"{log_path}/head_trajectory_e{epsilon}_A{A}_f{f}.png")

    pylog.info("Simulation completed for all parameter combinations.")

if __name__ == '__main__':
    exercise0()
    plt.show()
