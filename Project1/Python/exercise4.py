
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
from util.rw import load_object
import numpy as np
import matplotlib.pyplot as plt

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

def GradientDescent(lr=0.1, itermax=100, tolerance=0.01):
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    cpg_amplitude_gain = 0.125 * np.ones(13)
    errors = []

    simulate(0, cpg_amplitude_gain, log_path)
    controller = load_object(f'{log_path}controller0')
    A_res = controller.metrics["mech_joint_amplitudes"]
    error_joint_amp = np.linalg.norm(REF_JOINT_AMP - A_res)
    errors.append(error_joint_amp)

    iter = 1
    while error_joint_amp > tolerance and iter < itermax:
        # Prevent division by zero
        A_res_safe = np.clip(A_res[:-2], 1e-5, None)

        # Update gains with gradient step
        cpg_amplitude_gain = cpg_amplitude_gain * (1 + lr * (-1 + REF_JOINT_AMP[:-2] / A_res_safe))

        # Clamp values to keep them within reasonable physical range
        cpg_amplitude_gain = np.clip(cpg_amplitude_gain, 0.01, 2.0)

        try:
            simulate(iter, cpg_amplitude_gain, log_path)
            controller = load_object(f'{log_path}controller{iter}')
            A_res = controller.metrics["mech_joint_amplitudes"]

            if np.any(np.isnan(A_res)) or np.any(np.isinf(A_res)):
                print(f"Invalid values detected at iteration {iter}, stopping optimization.")
                break

            error_joint_amp = np.linalg.norm(REF_JOINT_AMP - A_res)
            errors.append(error_joint_amp)

        except Exception as e:
            print(f"Simulation failed at iteration {iter}: {e}")
            break

        if iter % 10 == 0:
            print(f"[Iteration {iter}] Current error: {error_joint_amp:.4f}")
        iter += 1

    print(f"Optimization ended at iteration {iter} with error = {error_joint_amp:.4f}")

    # Plot error evolution
    plt.figure()
    plt.plot(errors, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Error (L2 norm)")
    plt.title("Error Evolution During Optimization")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return cpg_amplitude_gain

def exercise4():
    print("Exercise 4")
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

def question_4_2():
    print("Running Question 4.2")
    pylog.info("Starting optimization for Question 4.2")

    # Log directory
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    # Run optimization
    cpg_amplitude_gain = GradientDescent(lr=0.1, itermax=100, tolerance=0.01)

    # Load the last controller based on file names
    try:
        controller_files = [f for f in os.listdir(log_path) if f.startswith("controller")]
        trial_indices = []

        for f in controller_files:
            try:
                idx = int(f.replace("controller", ""))
                trial_indices.append(idx)
            except ValueError:
                continue

        trial_idx = max(trial_indices)
        controller = load_object(f'{log_path}controller{trial_idx}')
        print(f"Loaded controller{trial_idx}")
    except Exception as e:
        print(f"Error loading the last valid controller: {e}")
        return

    A_res = controller.metrics["mech_joint_amplitudes"]

    # Plot joint amplitudes: reference vs optimized
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(REF_JOINT_AMP)), REF_JOINT_AMP, label='Reference', alpha=0.7)
    plt.bar(np.arange(len(A_res)), A_res, label='Optimized', alpha=0.7)
    plt.xlabel("Joint Index")
    plt.ylabel("Amplitude (rad)")
    plt.title("Joint Amplitudes: Reference vs Optimized")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_path}amplitude_comparison.png")
    plt.show()

    # Plot joint angles over time
    joint_angles = controller.states["joint_angles"]
    plt.figure(figsize=(12, 6))
    for i in range(joint_angles.shape[1]):
        plt.plot(joint_angles[:, i], label=f'Joint {i}')
    plt.xlabel("Time (iterations)")
    plt.ylabel("Joint Angle (rad)")
    plt.title("Evolution of Joint Angles Over Time")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_path}joint_angle_evolution.png")
    plt.show()

    print(f"Plots saved to: {log_path}")

    input("Press Enter to exit...")


def question_4_3():
    print("Running Question 4.3")
    pylog.info("Comparing metrics for Question 4.3")

    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    # Run non-optimized simulation
    print("Simulating non-optimized controller...")
    trial_unoptimized = "non_optimized"
    simulate(trial_unoptimized, 0.125 * np.ones(13), log_path)

    controller_unopt = load_object(f'{log_path}controller{trial_unoptimized}')
    metrics_unopt = controller_unopt.metrics

    # Load optimized simulation
    try:
        trial_idx = max([int(f.replace("controller", "")) for f in os.listdir(log_path) if "controller" in f and f.replace("controller", "").isdigit()])
        controller_opt = load_object(f'{log_path}controller{trial_idx}')
        metrics_opt = controller_opt.metrics
    except Exception as e:
        print(f"Could not load optimized controller: {e}")
        return

    # --- STEP 3: Display metrics comparison ---
    print("\n===== METRICS COMPARISON =====")
    def print_metric(metric_name):
        val_unopt = metrics_unopt.get(metric_name, "N/A")
        val_opt = metrics_opt.get(metric_name, "N/A")
        print(f"{metric_name:<30} | Non-Optimized: {val_unopt:<20} | Optimized: {val_opt}")

    key_metrics = [
        "mech_joint_amplitudes",
        "mech_power",
        "mech_cost_of_transport",
        "neural_energy",
        "neural_cost",
        "swim_speed",
    ]

    for key in key_metrics:
        print_metric(key)


if __name__ == '__main__':
    # Launch one of the questions
    # exercise4()       # Gradient descent
    question_4_2()      # Optimisation plus plot
    # question_4_3()    # Metrics comparison
    

