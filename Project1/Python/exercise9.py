import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from util.run_closed_loop import run_single, run_multiple
from util.rw import load_object
from simulation_parameters import SimulationParameters
from plotting_common import plot_time_histories, plot_time_histories_multiple_windows

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

NUM_PROCESS = 10


def test_cpg_configurations(config_name, weights_body2body, weights_contralateral, 
                           feedback_weights_range, log_path):
    """
    Test different CPG configurations with varying feedback weights
    """
    print(f"\n{'='*60}")
    print(f"Testing configuration: {config_name}")
    print(f"weights_body2body: {weights_body2body}")
    print(f"weights_body2body_contralateral: {weights_contralateral}")
    print(f"{'='*60}")
    
    # Test different feedback weights
    w_values = feedback_weights_range
    results = []
    
    pars_list = []
    for i, w in enumerate(w_values):
        pars_list.append(SimulationParameters(
            simulation_i=i,
            n_iterations=5001,
            log_path=log_path + config_name + '/',
            video_record=False,
            headless=True,
            controller="abstract oscillator",
            print_metrics=False,
            compute_metrics='all',
            cpg_amplitude_gain=REF_JOINT_AMP[:-2],
            # CPG coupling weights
            weights_body2body=weights_body2body,
            weights_body2body_contralateral=weights_contralateral,
            # Feedback weights (scaled by ws_ref)
            feedback_weights_ipsi=w * ws_ref,
            feedback_weights_contra=-w * ws_ref,
            # Initial perturbation to help initiate swimming
            initial_phases=np.random.uniform(-0.1, 0.1, 26),  # Small random initial phases
        ))
    
    # Run simulations
    run_multiple(pars_list, num_process=NUM_PROCESS)
    
    # Analyze results
    for i, w in enumerate(w_values):
        try:
            controller = load_object(log_path + config_name + '/' + f"controller{i}")
            
            freq = controller.metrics["neur_frequency"]
            speed = controller.metrics["mech_speed_fwd"]
            cot = controller.metrics["mech_cot"]
            amp = controller.metrics["mech_mean_amplitude"]
            
            # Check if swimming is successful (frequency > 0.5 Hz and forward speed > 0.01 m/s)
            swimming_success = freq > 0.5 and speed > 0.01 and not np.isnan(freq)
            
            results.append({
                'w': w,
                'freq': freq,
                'speed': speed,
                'cot': cot,
                'amplitude': amp,
                'success': swimming_success
            })
            
            print(f"w={w:.2f}: freq={freq:.3f} Hz, speed={speed:.4f} m/s, "
                  f"swimming={'YES' if swimming_success else 'NO'}")
        except Exception as e:
            print(f"w={w:.2f}: Simulation failed - {str(e)}")
            results.append({
                'w': w,
                'freq': np.nan,
                'speed': np.nan,
                'cot': np.nan,
                'amplitude': np.nan,
                'success': False
            })
    
    return results, pars_list


def plot_configuration_results(results, config_name, plot_dir):
    """
    Plot results for a specific configuration
    """
    # Filter out failed simulations
    valid_results = [r for r in results if not np.isnan(r['freq'])]
    
    if not valid_results:
        print(f"No valid results for {config_name}")
        return
    
    w_values = [r['w'] for r in valid_results]
    frequencies = [r['freq'] for r in valid_results]
    speeds = [r['speed'] for r in valid_results]
    amplitudes = [r['amplitude'] for r in valid_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Frequency
    ax1.plot(w_values, frequencies, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Swimming threshold')
    ax1.set_xlabel('Feedback gain w')
    ax1.set_ylabel('Neural Frequency [Hz]')
    ax1.set_title('Neural Frequency vs Feedback Gain')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Speed
    ax2.plot(w_values, speeds, 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Swimming threshold')
    ax2.set_xlabel('Feedback gain w')
    ax2.set_ylabel('Forward Speed [m/s]')
    ax2.set_title('Swimming Speed vs Feedback Gain')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Amplitude
    ax3.plot(w_values, amplitudes, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Feedback gain w')
    ax3.set_ylabel('Mean Amplitude [rad]')
    ax3.set_title('Joint Amplitude vs Feedback Gain')
    ax3.grid(True, alpha=0.3)
    
    # Success indicator
    all_w = [r['w'] for r in results]
    all_success = [r['success'] for r in results]
    ax4.scatter(all_w, all_success, s=100, c=['red' if not s else 'green' for s in all_success])
    ax4.set_xlabel('Feedback gain w')
    ax4.set_ylabel('Swimming Success')
    ax4.set_title('Swimming Success vs Feedback Gain')
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Configuration: {config_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{config_name}_results.png'), dpi=300)
    plt.close()


def run_detailed_comparison(success_w, success_result, config_name, log_path, plot_dir):
    """
    Run detailed comparison between normal CPG and CPG-free swimming
    """
    print(f"\nRunning detailed comparison for {config_name}")
    
    # Normal CPG parameters
    normal_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        headless=True,
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=30,
        weights_body2body_contralateral=10,
        feedback_weights_ipsi=1.0,
        feedback_weights_contra=-1.0,
        return_network=True,  # IMPORTANT: This was missing!
    )
    
    # CPG-free parameters
    if config_name == "no_ipsilateral":
        w_b2b, w_contra = 0, 10
    elif config_name == "no_contralateral":
        w_b2b, w_contra = 30, 0
    else:  # no_cpg
        w_b2b, w_contra = 0, 0
    
    cpg_free_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        compute_metrics='all',
        headless=True,
        cpg_amplitude_gain=REF_JOINT_AMP[:-2],
        weights_body2body=w_b2b,
        weights_body2body_contralateral=w_contra,
        feedback_weights_ipsi=success_w * ws_ref,
        feedback_weights_contra=-success_w * ws_ref,
        initial_phases=np.random.uniform(-0.1, 0.1, 26),
        return_network=True,  # IMPORTANT: This was missing!
        video_record=True,
        video_name=f"{config_name}_swimming",
        video_fps=50,
    )
    
    # Run simulations
    controller_normal = run_single(normal_pars)
    controller_cpg_free = run_single(cpg_free_pars)
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    times = controller_normal.times
    time_window = (times >= 2) & (times <= 4)
    
    # Motor outputs
    for i in range(0, 6, 2):
        joint_idx = i // 2
        motor_diff_normal = controller_normal.motor_out[time_window, i] - controller_normal.motor_out[time_window, i+1]
        motor_diff_cpg_free = controller_cpg_free.motor_out[time_window, i] - controller_cpg_free.motor_out[time_window, i+1]
        
        ax1.plot(times[time_window], motor_diff_normal, label=f'Joint {joint_idx}')
        ax2.plot(times[time_window], motor_diff_cpg_free, label=f'Joint {joint_idx}')
    
    ax1.set_title('Normal CPG Swimming')
    ax1.set_ylabel('Motor Output Difference (L-R)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title(f'{config_name.replace("_", " ").title()} Swimming')
    ax2.set_ylabel('Motor Output Difference (L-R)')
    ax2.legend()
    ax2.grid(True)
    
    # Joint angles
    for i in range(5):
        ax3.plot(times[time_window], controller_normal.joints_positions[time_window, i], label=f'Joint {i}')
        ax4.plot(times[time_window], controller_cpg_free.joints_positions[time_window, i], label=f'Joint {i}')
    
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Joint Angle [rad]')
    ax3.set_title('Normal CPG - Joint Angles')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Joint Angle [rad]')
    ax4.set_title(f'{config_name.replace("_", " ").title()} - Joint Angles')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{config_name}_comparison.png'), dpi=300)
    plt.close()
    
    # Print metrics comparison
    print(f"\nMetrics Comparison - {config_name}:")
    print(f"{'Metric':<25} {'Normal CPG':<15} {'CPG-Modified':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('Neural Frequency [Hz]', 'neur_frequency'),
        ('Forward Speed [m/s]', 'mech_speed_fwd'),
        ('Cost of Transport', 'mech_cot'),
        ('Mean Amplitude [rad]', 'mech_mean_amplitude'),
        ('Total Wave Lag', 'neur_twl')
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        val_normal = controller_normal.metrics[metric_key]
        val_cpg_free = controller_cpg_free.metrics[metric_key]
        diff = val_cpg_free - val_normal
        print(f"{metric_name:<25} {val_normal:<15.4f} {val_cpg_free:<15.4f} {diff:<15.4f}")


def exercise9():
    """
    Questions 7.1 and 7.2: CPG-free swimming experiments
    """
    pylog.info("Exercise 9 - Questions 7.1 and 7.2: CPG-free swimming")
    
    log_path = './logs/exercise9/'
    plot_dir = './plots/exercise9/'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define feedback weight range to test
    w_range = np.linspace(0, 2, 11)
    
    # Question 7.1: Remove ipsilateral connections
    print("\n" + "="*80)
    print("QUESTION 7.1: Removing ipsilateral connections (weights_body2body = 0)")
    print("="*80)
    
    results_no_ipsi, pars_no_ipsi = test_cpg_configurations(
        config_name="no_ipsilateral",
        weights_body2body=0,  # Remove ipsilateral connections
        weights_contralateral=10,  # Keep contralateral connections
        feedback_weights_range=w_range,
        log_path=log_path
    )
    
    plot_configuration_results(results_no_ipsi, "no_ipsilateral", plot_dir)
    
    # Question 7.2: Remove contralateral connections
    print("\n" + "="*80)
    print("QUESTION 7.2: Removing contralateral connections")
    print("="*80)
    
    results_no_contra, pars_no_contra = test_cpg_configurations(
        config_name="no_contralateral",
        weights_body2body=30,  # Keep ipsilateral connections
        weights_contralateral=0,  # Remove contralateral connections
        feedback_weights_range=w_range,
        log_path=log_path
    )
    
    plot_configuration_results(results_no_contra, "no_contralateral", plot_dir)
    
    # Question 7.2 continued: Remove both connections
    results_no_cpg, pars_no_cpg = test_cpg_configurations(
        config_name="no_cpg",
        weights_body2body=0,  # Remove ipsilateral connections
        weights_contralateral=0,  # Remove contralateral connections
        feedback_weights_range=w_range,
        log_path=log_path
    )
    
    plot_configuration_results(results_no_cpg, "no_cpg", plot_dir)
    
    # Find successful configurations
    success_no_ipsi = [(r['w'], r) for r in results_no_ipsi if r['success']]
    success_no_contra = [(r['w'], r) for r in results_no_contra if r['success']]
    success_no_cpg = [(r['w'], r) for r in results_no_cpg if r['success']]
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    print(f"\nConfiguration success summary:")
    print(f"- No ipsilateral: {len(success_no_ipsi)} successful out of {len(results_no_ipsi)}")
    print(f"- No contralateral: {len(success_no_contra)} successful out of {len(results_no_contra)}")
    print(f"- No CPG (both removed): {len(success_no_cpg)} successful out of {len(results_no_cpg)}")
    
    # Run detailed comparisons for successful cases
    if success_no_cpg:
        print(f"\n✓ CPG-FREE SWIMMING IS POSSIBLE!")
        print(f"  Minimum feedback gain needed: w = {success_no_cpg[0][0]:.2f}")
        print(f"  (Actual weights: ipsi={success_no_cpg[0][0]*ws_ref:.1f}, contra={-success_no_cpg[0][0]*ws_ref:.1f})")
        run_detailed_comparison(success_no_cpg[0][0], success_no_cpg[0][1], "no_cpg", log_path, plot_dir)
    else:
        print("\n✗ CPG-FREE SWIMMING NOT ACHIEVED with tested parameters")


if __name__ == '__main__':
    exercise9()