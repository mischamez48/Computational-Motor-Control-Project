
from plotting_common import plot_time_histories, save_figures
from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
import numpy as np
import matplotlib
matplotlib.rc('font', **{"size": 15})
from util.rw import load_object

num_process = 10  # number of processes to run in parallel
ylim_amp = [0, 0.01]


def question_2_2():
    """
    Run simulations to investigate the effect of different damping ratios (DR) on joint dynamics.
    
    This function simulates a zebrafish model with different damping factors and no controller input 
    (amp=0) to observe the passive joint dynamics. The simulations are run:
    - In zero gravity environment (gravity=np.zeros(3))
    - With an initial joint angle configuration of 0.3π for all joints
    - 10 meters above ground to eliminate water drag effects
    
    Parameters tested:
    - Damping ratios (DR): [1.0, 0.3, 0.1] representing overdamped, critically damped,
      and underdamped configurations respectively
    
    """
    prepath = './logs/exercise1/question_2_2/'
    DR = [1, 0.3, 0.1]
    animal_pose = [0.0, 0.0, 10.0, 0.0, 0, -1.570796327] #10 meters above ground
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=1001,
            log_path= prepath,
            damping_factor=dr,
            video_record=False,
            headless=True,
            amp = 0,
            controller="empty",
            gravity=np.zeros(3),
            joint_poses=0.3*np.ones(15), # joint angles at 0.3pi
            animal_pose = animal_pose,
            print_metrics=False,
            compute_metrics='mechanical',
        )
        for i, dr in enumerate(DR)
    ]

    run_multiple(pars_list, num_process=num_process)
    save_dir = './plots/exercise1/question_2_2/'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(DR)):
        # load controller
        controller = load_object(prepath+"controller"+str(i))
        joint_angles = controller.joints_positions
        n_joints = joint_angles.shape[1]
        joint_labels = [f"Joint {i}" for i in range(n_joints)]
        plot_time_histories(controller.times, 
                            joint_angles, 
                            labels=joint_labels, 
                            title=f"Joint Angles (DR={DR[i]})", 
                            closefig=False, 
                            savepath=save_dir + " Joint Angles_DR_"+str(DR[i]).replace('.', '_'))
        plt.grid(True)
        plt.show()
    
def question_2_3(animal_pose, twl=0):
    """
    Runs a systematic study of the effect of the wave controller frequency 
    on joint amplitudes for different damping ratios.
    
    Parameters:
    - animal_pose: List with position coordinates - determines if in water or space
    - twl: Total wave lag value (default 0)
    - savename: Filename suffix for saving the plot

    """
    prepath = './logs/exercise1/question_2_3/'
    nsim = 20

    DR = [1, 0.3, 0.1]
    frequencies = np.linspace(0, 20, nsim)
    amp = 0.01
    
    # Determine environment type for the title
    environment = "Space" if animal_pose[2] > 0 else "Water"

    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=1001,
            log_path=prepath,
            damping_factor=dr,
            video_record=False,
            headless=True,
            controller="sine",  # Using wave controller
            amp=amp,
            twl=twl,
            freq=freq,
            gravity=np.zeros(3),
            joint_poses=0.3*np.ones(15),  # joint angles at 0.3pi
            animal_pose=animal_pose,
            print_metrics=False,
            compute_metrics='mechanical',
        )
        for i, dr in enumerate(DR)
        for j, freq in enumerate(frequencies)
    ]
    run_multiple(pars_list, num_process=num_process)

    DR_cases_list = []
    for i in range(len(DR)):
        mean_lists = []
        for j in range(nsim):
            controller = load_object(prepath+"controller"+str(i*nsim+j))
            mean_amplitude = controller.metrics["mech_joint_amplitudes"].mean()
            mean_lists.append(mean_amplitude)

        DR_cases_list.append(mean_lists)

    DR_cases = np.array(DR_cases_list).T

    labels = [f"DR = {dr}" for dr in DR]
    save_dir = './plots/exercise1/question_2_3/'
    os.makedirs(save_dir, exist_ok=True)
    
    plot_time_histories(
        frequencies,
        DR_cases,
        labels=labels,
        title=f"Mean Joint Amplitude vs Frequency (TWL={twl}, {environment})",
        ylabel="Mean Joint Amplitude [rad]",
        xlabel="Frequency [Hz]",
        closefig=False,  # FALSE = Don't close the figure yet
        savepath=f"{save_dir}Mean_Joint_Amplitude_vs_Frequency_TWL_{twl}_{environment}.png"
    )
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def question_2_4(muscle_parameters_tag):
    """
    Studies the relationship between activation frequency and total wave lag (TWL)
    for a specific muscle parameter configuration.
    
    Parameters:
    - muscle_parameters_tag: String identifier for the muscle parameter set to use

    """
    prepath = './logs/exercise1/question_2_4/'

    frequencies = np.arange(3, 41, 2)  # frequencies from 3 to 40 Hz
    nsim = frequencies.shape[0]
    TWL = np.arange(0.0, 2.1, 0.2)      # TWL values from 0.0 to 2.0
    amp = 0.5

    # Extract resonant frequency from muscle parameter tag for reference
    # Format is FN_XXXX where XXXX is frequency in 0.1Hz (e.g., 5000 means 5Hz)
    try:
        resonant_freq = float(muscle_parameters_tag.split('_')[1])/1000
    except:
        resonant_freq = None

    pars_list = [
        SimulationParameters(
            simulation_i=i + j * nsim,
            n_iterations=1001,
            log_path=prepath,
            amp=amp,
            freq=freq,
            twl=twl,
            video_record=False,
            headless=True,
            controller="sine",  # FIXED: Use sine wave controller instead of empty
            gravity=np.zeros(3),
            joint_poses=0.3 * np.ones(15),  # joint angles at 0.3*pi
            muscle_parameters_tag=muscle_parameters_tag,
            print_metrics=False,
            compute_metrics='mechanical',
        )
        for j, twl in enumerate(TWL)
        for i, freq in enumerate(frequencies)
    ]

    run_multiple(pars_list, num_process=num_process)

    # Collect simulation results for each TWL case across all frequencies
    TWL_speed_list = []
    TWL_cot_list = []
    for j in range(len(TWL)):      # Outer loop over TWL values
        speed_list = []
        cot_list = []
        for i in range(nsim):      # Inner loop over frequency values
            controller = load_object(prepath + "controller" + str(j * nsim + i))
            speed = controller.metrics["mech_speed_fwd"]
            cot = controller.metrics["mech_cot"]
            speed_list.append(speed)
            cot_list.append(cot)
        TWL_speed_list.append(speed_list)
        TWL_cot_list.append(cot_list)

    TWL_speeds = np.array(TWL_speed_list).T
    TWL_cot = np.array(TWL_cot_list).T

    labels = [f"TWL = {twl:.1f}" for twl in TWL]
    save_dir = './plots/exercise1/question_2_4/'
    os.makedirs(save_dir, exist_ok=True)

    plot_time_histories(
        frequencies,
        TWL_speeds,
        labels=labels,
        title=f"Forward Speed vs Frequency ({muscle_parameters_tag})",
        ylabel="Forward Speed [m/s]",
        xlabel="Frequency [Hz]",
        closefig=False,
        savepath=save_dir+muscle_parameters_tag
    )
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add vertical line at resonant frequency if available
    if resonant_freq:
        plt.axvline(x=resonant_freq, color='red', linestyle='--', alpha=0.7)
        plt.text(resonant_freq, plt.ylim()[1]*0.9, f"ω_r={resonant_freq}Hz", 
                 rotation=90, verticalalignment='top', color='red')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Cost of transport (CoT).
    # The cost of transport relates the energy expenditure to the distance traveled.
    # CoT = E/D_fwd, where D_fwd is the forward distance covered by the center of mass.

    plot_time_histories(
        frequencies,
        TWL_cot,
        labels=labels,
        title=f"Cost of Transport vs Frequency ({muscle_parameters_tag})",
        ylabel="Cost of Transport [J/m]",
        xlabel="Frequency [Hz]",
        closefig=False,
        savepath=save_dir+muscle_parameters_tag+"_cot"
    )
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add vertical line at resonant frequency if available
    if resonant_freq:
        plt.axvline(x=resonant_freq, color='red', linestyle='--', alpha=0.7)
        plt.text(resonant_freq, plt.ylim()[1]*0.9, f"ω_r={resonant_freq}Hz", 
                 rotation=90, verticalalignment='top', color='red')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Optional: Create a 2D heatmap visualization of speed vs frequency and TWL
    speed_2d = TWL_speeds.T  # Reshape for 2D plot (rows=TWL, cols=frequency)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(frequencies, TWL, speed_2d, shading='auto', cmap='viridis')
    plt.colorbar(label='Forward Speed [m/s]')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Total Wave Lag')
    plt.title(f'Forward Speed Heatmap ({muscle_parameters_tag})')
    plt.tight_layout()
    plt.savefig(save_dir+muscle_parameters_tag+"_heatmap")
    plt.show()

def exercise1():

    pylog.info("Ex 1")
    pylog.info("Implement exercise 1")
    prepath = './logs/exercise1/'
    
    # Initialization of the simulation parameters
    animal_pose = [0.0, 0.0, 10.0, 0.0, 0, -1.570796327] # Should be 10 meters above ground but error in the pdf
    animal_pose_default = [0.0, 0.0, -0.01, 0.0, 0, -1.570796327]
    damping_factor = 1 # damping factor for the simulation

    # Question 2.2
    question_2_2()
    

    # Question 2.3
    question_2_3(animal_pose_default, twl=0)
    question_2_3(animal_pose, twl=0)
    question_2_3(animal_pose_default, twl=0.5)
    question_2_3(animal_pose, twl=0.5)

    # Question 2.4
    muscle_parameters_tags = ["FN_5000_ZC_1000_G0_419",
                              "FN_7500_ZC_1000_G0_419",
                              "FN_10000_ZC_1000_G0_419"]
    
    for muscle_parameters_tag in muscle_parameters_tags:
        question_2_4(muscle_parameters_tag)
    


if __name__ == '__main__':

    exercise1()
