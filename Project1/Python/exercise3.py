"""
Exercise 3: Implementation of a CPG network for zebrafish swimming simulation.

This file contains functions to:
1. Run a basic test of CPG implementation (Question 3.2)
2. Generate and analyze plots of oscillator behavior (Question 3.3)
3. Explore the effect of drive parameter on swimming performance (Question 3.4)
"""

from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import matplotlib.pyplot as plt
from plotting_common import plot_time_histories, plot_time_histories_multiple_windows
import numpy as np
from util.rw import load_object

# Number of processes for parallel simulation runs
num_process = 10

def question_3_2():
    """
    Test the basic implementation of the CPG network.
    
    This function runs a simple simulation to verify that your implementation
    in abstract_oscillator_controller.py is working correctly.
    """
    log_path = './logs/exercise3/question_3_2/'
    os.makedirs(log_path, exist_ok=True)
    
    # Basic parameters for testing implementation
    all_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        log_path=log_path,
        compute_metrics="mechanical",  # Compute mechanical metrics to verify swimming
        print_metrics=True,            # Print metrics to console for verification
        headless=False,                # Set to True for running without GUI
        # CPG parameters
        drive=4,
        cpg_frequency_gain=0.6,
        cpg_frequency_offset=0.6,
        cpg_amplitude_gain=0.125 * np.ones(13),
        weights_body2body=30,
        weights_body2body_contralateral=10,
        phase_lag_body=2*np.pi,
        amplitude_rates=1,
        motor_output_scaling=1,
    )
    
    pylog.info("Running Question 3.2: Testing CPG implementation")
    controller = run_single(all_pars)
    
    pylog.info("Simulation completed. Check the GUI to verify swimming motion.")
    
    return controller

def question_3_3():
    """
    Generate and analyze plots of the CPG network behavior.
    
    This function runs a simulation and generates plots of oscillator phases,
    amplitudes, motor outputs, and joint angles over time.
    """
    log_path = './logs/exercise3/question_3_3/'
    os.makedirs(log_path, exist_ok=True)
    save_dir = './plots/exercise3/question_3_3/'
    os.makedirs(save_dir, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        log_path=log_path,
        compute_metrics=None,
        print_metrics=False,
        return_network=True,
        initial_phases=0, 
        # CPG parameters
        drive=1,
        cpg_frequency_gain=0.6,
        cpg_frequency_offset=0.6,
        cpg_amplitude_gain=0.125*np.ones(13),
        weights_body2body=300,
        weights_body2body_contralateral=100,
        phase_lag_body=2*np.pi,
        amplitude_rates=1,
        motor_output_scaling=0.5,
        # Additional settings
        video_record=True,             # Set to True to record a video
        video_name="zebrafish_cpg",    # Name of the output video file
        headless=False,                # Set to True for running without GUI
    )

    pylog.info("Running Question 3.3: Generating CPG behavior plots")
    controller = run_single(all_pars)

    # Extract data from the controller
    phases = controller.state[:, :2*all_pars.n_joints]
    amplitudes = controller.state[:, 2*all_pars.n_joints:]
    motor_l = controller.motor_out[:, controller.motor_l]
    motor_r = controller.motor_out[:, controller.motor_r]
    motor_diff = motor_l - motor_r
    joint_angles = controller.joints_positions

    # Print data shapes for debugging
    pylog.info(f"Phases shape: {phases.shape}")
    pylog.info(f"Amplitudes shape: {amplitudes.shape}")
    pylog.info(f"Left motor outputs shape: {motor_l.shape}")
    pylog.info(f"Right motor outputs shape: {motor_r.shape}")
    pylog.info(f"Motor difference shape: {motor_diff.shape}")
    pylog.info(f"Joint angles shape: {joint_angles.shape}")

    # Generate and save plots
    plt.figure("Oscillator Phases")
    plot_time_histories(controller.times, phases, ylabel="Phase (rad)", 
                        closefig=True, savepath=save_dir+"phases")

    plt.figure("Oscillator Amplitudes")
    plot_time_histories(controller.times, amplitudes, ylabel="Amplitude", 
                        closefig=True, savepath=save_dir+"amplitudes")
    
    plt.figure("Left Motor Outputs")
    plot_time_histories(controller.times, motor_l, ylabel="Activation", 
                        closefig=True, savepath=save_dir+"motor_l")

    plt.figure("Right Motor Outputs")
    plot_time_histories(controller.times, motor_r, ylabel="Activation", 
                        closefig=True, savepath=save_dir+"motor_r")

    plt.figure("Motor Activation Difference (Left - Right)")
    plot_time_histories(controller.times, motor_diff, ylabel="Difference", 
                        closefig=True, savepath=save_dir+"motor_diff")

    plt.figure("Joint Angles")
    plot_time_histories(controller.times, joint_angles, ylabel="Angle (rad)", 
                        closefig=True, savepath=save_dir+"joint_angles")
    
    plt.show()
    
    return controller

def question_3_4():
    """
    Explore the effect of the drive parameter on swimming performance.
    
    This function runs multiple simulations with different drive values and
    analyzes how changing the drive affects swimming metrics.
    """
    # Define range of drive values to test
    drives = np.linspace(1, 12, 11).astype(np.int8)
    
    log_path = './logs/exercise3/question_3_4/'
    os.makedirs(log_path, exist_ok=True)
    save_dir = './plots/exercise3/question_3_4/'
    os.makedirs(save_dir, exist_ok=True)

    # Create parameter sets for each drive value
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=5001,
            log_path=log_path,
            video_record=False,
            headless=True,
            controller="abstract oscillator",
            print_metrics=False,
            compute_metrics='mechanical',
            # CPG parameters with varying drive
            drive=drive,
            cpg_frequency_gain=0.6,
            cpg_frequency_offset=0.6,
            cpg_amplitude_gain=0.125*np.ones(13),
            weights_body2body=30,
            weights_body2body_contralateral=10,
            phase_lag_body=2*np.pi,
            amplitude_rates=1,
            motor_output_scaling=1.0,
        )
        for i, drive in enumerate(drives)
    ]

    pylog.info("Running Question 3.4: Exploring drive parameter effects")
    # Run multiple simulations in parallel
    run_multiple(pars_list, num_process=num_process)

    # Lists to store metrics from all simulations
    cot_list = []           # Cost of transport
    speed_list = []         # Forward swimming speed
    mech_mean_freq_list = [] # Mechanical oscillation frequency
    mech_mean_amp_list = []  # Mechanical oscillation amplitude

    # Process results for each drive value
    for i, drive in enumerate(drives):
        # Load controller from saved file
        controller = load_object(log_path + "controller" + str(i))
        
        # Extract motor outputs
        motor_output = controller.motor_out
        
        # Plot left motor outputs for this drive value
        plot_time_histories(
            controller.times, 
            motor_output[:, controller.motor_l], 
            title=f"Left Motor Outputs (Drive = {drive})", 
            ylabel="Activation",
            closefig=True, 
            savepath=save_dir + f"drive_{drive}"
        )
        
        # Extract metrics
        cot = controller.metrics["mech_cot"]
        speed = controller.metrics["mech_speed_fwd"]
        mech_mean_freq = controller.metrics["mech_mean_frequency"]
        mech_mean_amp = controller.metrics["mech_mean_amplitude"]

        # Store metrics
        cot_list.append(cot)
        speed_list.append(speed)
        mech_mean_freq_list.append(mech_mean_freq)
        mech_mean_amp_list.append(mech_mean_amp)

    # Convert lists to arrays for plotting
    cot = np.array(cot_list)
    speed = np.array(speed_list)
    mech_mean_freq = np.array(mech_mean_freq_list)
    mech_mean_amp = np.array(mech_mean_amp_list)

    # Combine all metrics into a single array
    states = np.column_stack((cot, speed, mech_mean_freq, mech_mean_amp))
    labels = ["Cost of Transport", "Swimming Speed", "Mechanical Frequency", "Mechanical Amplitude"]
    
    # Plot all metrics vs drive
    plt.figure("Locomotion Performance Metrics vs Drive")
    plot_time_histories_multiple_windows(
        drives, 
        states, 
        labels=labels, 
        xlabel="Drive Parameter", 
        ylabel=None, 
        closefig=True,
        savepath=save_dir + "metrics"
    )

    plt.show()
    
    # Print summary for report
    pylog.info("\nSummary of drive parameter effects:")
    pylog.info("Drive values: " + ", ".join(str(d) for d in drives))
    pylog.info("Speed values: " + ", ".join(f"{s:.4f}" for s in speed))
    pylog.info("Cost of Transport: " + ", ".join(f"{c:.4f}" for c in cot))
    pylog.info("Mechanical frequencies: " + ", ".join(f"{f:.4f}" for f in mech_mean_freq))
    pylog.info("Mechanical amplitudes: " + ", ".join(f"{a:.4f}" for a in mech_mean_amp))

def exercise3(question=None):
    """
    Main function to run Exercise 3 questions.
    
    Args:
        question: String indicating which question to run.
                 Options: '3.2', '3.3', '3.4', or None to run all.
    """
    pylog.info("Starting Exercise 3: CPG Network Implementation")
    
    if question == '3.2' or question is None:
        question_3_2()
        
    if question == '3.3' or question is None:
        question_3_3()
        
    if question == '3.4' or question is None:
        question_3_4()

if __name__ == '__main__':
    # Choose which question to run by uncommenting the appropriate line
    # exercise3('3.2')  # Run only Question 3.2
    # exercise3('3.3')  # Run only Question 3.3
    exercise3('3.4')  # Run only Question 3.4