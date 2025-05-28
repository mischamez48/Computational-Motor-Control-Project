import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from simulation_parameters import SimulationParameters
from util.run_open_loop import run_single
from util.entraining_signals import define_entraining_signals
from plotting_common import plot_time_histories, plot_1d

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def question_6_1():
    """
    Question 6.1: Explore the entrainment by running exercise7.py, where a default 
    entrainment of 45 degrees at 8 Hz is implemented. Report the neural frequency of 
    the controller. How does the frequency compare to the simulation without entrainment? 
    Does this match your expectation and why?
    """
    
    log_path = './logs/exercise7/question_6_1/'
    os.makedirs(log_path, exist_ok=True)
    
    # Common parameters
    common_params = {
        'n_iterations': 5001,
        'controller': "abstract oscillator",
        'compute_metrics': 'all',
        'log_path': log_path,
        'print_metrics': False,
        'return_network': True,
        'headless': True,
        'cpg_amplitude_gain': REF_JOINT_AMP[:-2],  # Only active joints
        'feedback_weights_ipsi': 0.5,   # Some feedback to enable entrainment
        'feedback_weights_contra': -0.5,
    }
    
    # 1. Run simulation WITHOUT entrainment (baseline)
    pylog.info("Running simulation WITHOUT entrainment (baseline)")
    pars_no_entrain = SimulationParameters(
        **common_params,
        entraining_signals=None,  # No entrainment
        simulation_i=0,
    )
    controller_no_entrain = run_single(pars_no_entrain)
    freq_no_entrain = controller_no_entrain.metrics["neur_frequency"]
    
    # 2. Run simulation WITH entrainment (45 degrees at 8 Hz)
    pylog.info("Running simulation WITH entrainment (45 degrees at 8 Hz)")
    entraining_signals = define_entraining_signals(
        n_iterations=5001, 
        frequency=8.0,  # 8 Hz entrainment
        amplitude_degrees=45, 
        plot_signals=False
    )
    
    pars_entrain = SimulationParameters(
        **common_params,
        entraining_signals=entraining_signals,
        simulation_i=1,
    )
    controller_entrain = run_single(pars_entrain)
    freq_entrain = controller_entrain.metrics["neur_frequency"]
    
    # 3. Analysis and comparison
    print("\n" + "="*60)
    print("QUESTION 6.1: Entrainment Analysis Results")
    print("="*60)
    print(f"Neural frequency WITHOUT entrainment: {freq_no_entrain:.3f} Hz")
    print(f"Neural frequency WITH entrainment (8 Hz): {freq_entrain:.3f} Hz")
    print(f"Entrainment frequency: 8.0 Hz")
    print(f"Frequency shift: {freq_entrain - freq_no_entrain:.3f} Hz")
    print(f"Relative change: {((freq_entrain - freq_no_entrain) / freq_no_entrain * 100):.1f}%")
    print("="*60)
    
    # Plot oscillator outputs over time to visualize entrainment
    plot_dir = './plots/exercise7/question_6_1/'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot comparison of motor outputs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot first few oscillators for clarity
    n_osc_plot = 6
    times_no_entrain = controller_no_entrain.times
    times_entrain = controller_entrain.times
    
    # Without entrainment
    ax1.set_title(f'Motor Output WITHOUT Entrainment (f = {freq_no_entrain:.2f} Hz)')
    for i in range(0, n_osc_plot, 2):  # Plot every other oscillator
        motor_diff = controller_no_entrain.motor_out[:, i] - controller_no_entrain.motor_out[:, i+1]
        ax1.plot(times_no_entrain, motor_diff, label=f'Joint {i//2}')
    ax1.set_ylabel('Motor Output Difference (L-R)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim([3, 5])  # Zoom in to see oscillations clearly
    
    # With entrainment
    ax2.set_title(f'Motor Output WITH 8 Hz Entrainment (f = {freq_entrain:.2f} Hz)')
    for i in range(0, n_osc_plot, 2):
        motor_diff = controller_entrain.motor_out[:, i] - controller_entrain.motor_out[:, i+1]
        ax2.plot(times_entrain, motor_diff, label=f'Joint {i//2}')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Motor Output Difference (L-R)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim([3, 5])  # Zoom in to see oscillations clearly
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}motor_output_comparison.png", dpi=300)
    plt.close()
    
    return freq_no_entrain, freq_entrain


def exercise7():
    """
    Main function for exercise 7 - Question 6.1
    """
    pylog.info("Exercise 7 - Question 6.1: Entrainment Analysis")
    question_6_1()


if __name__ == '__main__':
    exercise7()