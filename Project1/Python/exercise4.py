from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
from util.rw import load_object
import numpy as np

# Reference amplitudes for 15 joints (last two passive)
REF_JOINT_AMP = np.array([
    0.06580, 0.02810, 0.02781, 0.03047, 0.03623,
    0.04127, 0.04864, 0.05398, 0.06508, 0.08945,
    0.10271, 0.11789, 0.14929, 0.0, 0.0
])
# Only the first 13 joints are active
REF13 = REF_JOINT_AMP[:13]

# Gradient descent function with momentum
# This function optimizes the CPG amplitude gain using gradient descent with momentum
# Momentum to break the plateau near the 46% of error
def GradientDescent(lr=0.1, itermax=100, tolerance=0.01, momentum=0.9):
    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    # Initialize gains
    cpg_amplitude_gain = 0.125 * np.ones(13)
    errors = []
    prev_update = np.zeros_like(cpg_amplitude_gain)  # Initialize momentum term

    # Initial simulation and error
    simulate(0, cpg_amplitude_gain, log_path)
    controller = load_object(f'{log_path}controller0')
    A_res = controller.metrics.get("mech_joint_amplitudes", None)
    if A_res is None:
        raise RuntimeError("No 'mech_joint_amplitudes' in metrics for controller0")
    A_res13 = np.array(A_res[:13])
    error_joint_amp = np.linalg.norm(REF13 - A_res13) / np.linalg.norm(REF13)
    errors.append(error_joint_amp)
    print(f"[Init] shapes REF13={REF13.shape}, A_res13={A_res13.shape}, error={error_joint_amp:.4f}")
    print(f"REF13:\n{REF13}")
    print(f"A_res13 (initial):\n{A_res13}")
    print(f"Ratio REF13/A_res13:\n{REF13 / np.clip(A_res13, 1e-5, None)}")

    iteration = 1
    while error_joint_amp > tolerance and iteration < itermax:
        # Compute the learning rate based on the iteration
        lr_eff = lr / np.sqrt(iteration)

        # Use a constant learning rate for simplicity
        # lr_eff = lr

        # Prevent division by zero
        A_safe = np.clip(A_res13, 1e-5, None)

        # Compute basic update
        raw_update = cpg_amplitude_gain * lr_eff * (REF13 / A_safe - 1)

        # Apply momentum
        update = momentum * prev_update + (1 - momentum) * raw_update
        prev_update = update.copy()

        # Update cpg_amplitude_gain
        cpg_amplitude_gain = np.clip(cpg_amplitude_gain + update, 0.01, 2.0)

        try:
            simulate(iteration, cpg_amplitude_gain, log_path)
            controller = load_object(f'{log_path}controller{iteration}')
            A_res = controller.metrics.get("mech_joint_amplitudes", None)
            if A_res is None:
                print(f"No amplitudes at iteration {iteration}, stopping.")
                break
            A_res13 = np.array(A_res[:13])

            if np.any(np.isnan(A_res13)) or np.any(np.isinf(A_res13)):
                print(f"Invalid amplitude values at iteration {iteration}, stopping.")
                break

            error_joint_amp = np.linalg.norm(REF13 - A_res13) / np.linalg.norm(REF13)
            errors.append(error_joint_amp)
            if iteration % 10 == 0:
                print(f"[Iter {iteration}] error={error_joint_amp:.4f}, lr_eff={lr_eff:.4f}")
        except Exception as e:
            print(f"Simulation error at iteration {iteration}: {e}")
            break

        iteration += 1

    print(f"Optimization ended at iter {iteration} with normalized error = {error_joint_amp:.4f}")

    # Plot error evolution
    plt.figure()
    plt.plot(errors, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Error")
    plt.title("Error Evolution During Optimization (with Momentum)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_path}error_evolution.png")
    plt.close()

    return cpg_amplitude_gain


def simulate(trial, cpg_amplitude_gain, log_path):
    all_pars = SimulationParameters(
        log_path=log_path,
        simulation_i=trial,
        headless=True,
        compute_metrics="all",
        print_metrics=False,
        controller="abstract oscillator",
        n_iterations=5001,
        drive=10,
        cpg_amplitude_gain=cpg_amplitude_gain,
    )
    _ = run_single(all_pars)


def question_4_2():
    print("Running Question 4.2 New Version")
    pylog.info("Starting optimization for Question 4.2")

    log_path = './logs/exercise4/'
    os.makedirs(log_path, exist_ok=True)

    cpg_amplitude_gain = GradientDescent(lr=0.2, itermax=200, tolerance=0.01, momentum=0.1)

    # Load last valid controller
    files = [f for f in os.listdir(log_path) if f.startswith('controller')]
    indices = [int(f.replace('controller','')) for f in files if f.replace('controller','').isdigit()]
    trial_idx = max(indices) if indices else None
    if trial_idx is None:
        print("No controller found, aborting question_4_2.")
        return
    controller = load_object(f'{log_path}controller{trial_idx}')
    print(f"Loaded controller{trial_idx}")

    A_res = controller.metrics.get("mech_joint_amplitudes", [])
    A_res13 = np.array(A_res[:13]) if len(A_res)>=13 else np.array(A_res)

    # Plot amplitudes comparison
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(13), REF13, label='Reference', alpha=0.7)
    plt.bar(np.arange(len(A_res13)), A_res13, label='Optimized', alpha=0.7)
    plt.xlabel("Joint index")
    plt.ylabel("Amplitude (rad)")
    plt.title("Joint amplitudes: Ref vs Optimized")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_path}amplitude_comparison.png")
    plt.close()

    # Plot joint angles time series
    # Plot joint angles time series
    if hasattr(controller, 'joints_positions'):
        joint_angles = controller.joints_positions
    else:
        print("Warning: No joint_positions found in controller. Skipping joint angle plot.")
        return
    plt.figure(figsize=(12,6))
    for i in range(joint_angles.shape[1]):
        plt.plot(joint_angles[:,i], label=f'Joint {i}')
    plt.xlabel("Time (steps)")
    plt.ylabel("Angle (rad)")
    plt.title("Joint angles over time")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_path}joint_angles_time.png")
    plt.close()

    print(f"Plots saved to {log_path}")


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

    def safe_get(metrics, key):
        return metrics.get(key, "N/A")

    def format_metric(val):
        if isinstance(val, (float, int)):
            return f"{val:.4f}"
        elif isinstance(val, (list, np.ndarray)):
            return f"mean={np.mean(val):.4f}"
        else:
            return str(val)

    key_metrics = [
        "mech_speed_fwd",
        "mech_cot",
        "mech_energy",
        "mech_torque",
        "mech_mean_frequency",
        "neur_frequency",
        "mech_joint_amplitudes",
    ]

    for key in key_metrics:
        val_unopt = safe_get(metrics_unopt, key)
        val_opt = safe_get(metrics_opt, key)
        print(f"{key:<25} | Non-Optimized: {format_metric(val_unopt):<20} | Optimized: {format_metric(val_opt)}")


def visualize_optimized_controller():
    from util.run_closed_loop import run_single
    from simulation_parameters import SimulationParameters
    import os
    from util.rw import load_object
    import numpy as np

    log_path = './logs/exercise4/'
    files = [f for f in os.listdir(log_path) if f.startswith('controller')]
    indices = [int(f.replace('controller','')) for f in files if f.replace('controller','').isdigit()]
    trial_idx = max(indices) if indices else None
    if trial_idx is None:
        print("No controller found.")
        return

    controller = load_object(f'{log_path}controller{trial_idx}')
    
    # Use a default if cpg_amplitude_gain is missing
    try:
        cpg_amplitude_gain = controller.cpg_amplitude_gain
    except AttributeError:
        # Otherwise, fallback: use the final mechanical amplitudes ratio
        mech_amp = controller.metrics.get("mech_joint_amplitudes", None)
        if mech_amp is None:
            raise RuntimeError("No mech_joint_amplitudes found to reconstruct cpg_amplitude_gain.")
        mech_amp = np.array(mech_amp[:13])
        # Fallback approximation: original gain * (target/reference)
        cpg_amplitude_gain = 0.125 * (REF13 / np.clip(mech_amp, 1e-5, None))

    # Rerun simulation with visualization
    all_pars = SimulationParameters(
        log_path=log_path,
        simulation_i="visu",
        headless=False,  # ATTENTION: interface graphique ouverte
        compute_metrics="all",
        print_metrics=True,
        controller="abstract oscillator",
        n_iterations=5001,  # ~5 seconds
        drive=8,
        cpg_amplitude_gain=cpg_amplitude_gain
    )
    run_single(all_pars)

if __name__ == '__main__':
    # Launch one of the questions
    # exercise4()       # Gradient descent
    # question_4_2()      # Optimisation plus plot
    question_4_3()    # Metrics comparison
    # visualize_optimized_controller()  # Visualize the optimized controller
    

