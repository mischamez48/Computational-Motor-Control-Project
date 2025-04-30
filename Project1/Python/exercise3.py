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
        initial_phases = 0, 
        drive = 1,
        cpg_frequency_gain = 0.6,
        cpg_frequency_offset = 0.6,
        cpg_amplitude_gain = 0.125*np.ones(13),
        weights_body2body = 300,
        weights_body2body_contralateral = 100,
        phase_lag_body = 2*np.pi,
        amplitude_rates = 1,
        motor_output_scaling = .5,
        # headless=False,
    )

    controller = run_single(
        all_pars
    )

    save_dir = './plots/exercise3/question_3_3/'
    os.makedirs(save_dir, exist_ok=True)
    phases = controller.state[:, :2*all_pars.n_joints]

    amplitudes = controller.state[:, 2*all_pars.n_joints:]
    motor_l = controller.motor_out[:, controller.motor_l]
    motor_r = controller.motor_out[:, controller.motor_r]
    motor_diff = motor_l - motor_r
    joint_angles = controller.joints_positions
    plt.figure("phases")
    plot_time_histories(controller.times, phases, ylabel=None, closefig= True, savepath = save_dir+"phases")

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
    # states = np.concatenate([cot_list, speed_list, mech_freq_list, mech_amp_list], axis=1)
    labels = ["CoT", "Speed", "Mech mean freq", "Mech mean amp"]
    plt.figure("locomotion performance")
    plot_time_histories_multiple_windows(drives, 
                                         states, 
                                         labels=labels, 
                                         xlabel="Drive", 
                                         ylabel=None, 
                                         closefig = True,
                                         savepath = save_dir + "metrics")

    plt.show()

def exercise3():

    pylog.info("Ex 3")
    log_path = './logs/exercise3/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        log_path=log_path,
        compute_metrics=None,
        print_metrics=False,
        return_network=True,
        headless=False,
        video_name="exercise3",
        video_record=True,
    )

    pylog.info("Running the simulation")
    # controller = run_single(
    #     all_pars
    # )

    question_3_3()
    question_3_4()
    
    # Hint: Optionally you can use some helper function to generate the plots
    # such as (plot_time_histories)


if __name__ == '__main__':
    # Choose which question to run by uncommenting the appropriate line
    # exercise3('3.2')  # Run only Question 3.2
    # exercise3('3.3')  # Run only Question 3.3
    exercise3('3.4')  # Run only Question 3.4