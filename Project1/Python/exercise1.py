
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

    # Should remove the wave controller term for this one?

    prepath = './logs/exercise1/question_2_2/'
    DR = [1, 0.3, 0.1]
    animal_pose = [0.0, 0.0, 0.1, 0.0, 0, -1.570796327]
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
                            title="Joint Angles", 
                            closefig=True, 
                            savepath=save_dir + "DR_"+str(DR[i]).replace('.', '_'))
        plt.show()
    
def question_2_3(animal_pose, twl = 0, savename="" ):
    prepath = './logs/exercise1/question_2_3/'
    nsim = 20

    DR = [1, 0.3, 0.1]
    frequencies = np.linspace(0, 20, nsim)
    amp=0.01

    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=1001,
            log_path= prepath,
            damping_factor=dr,
            video_record=False,
            headless=True,
            controller="empty",
            amp=amp,
            twl=twl,
            freq=freq,
            gravity=np.zeros(3),
            joint_poses=0.3*np.ones(15), # joint angles at 0.3pi
            animal_pose = animal_pose,
            print_metrics=False,
            compute_metrics='mechanical',
        )
        for i, dr in enumerate(DR)
        for j, freq in enumerate(np.linspace(0, 20, nsim))
    ]
    run_multiple(pars_list, num_process=num_process)

    DR_cases_list = []
    for i in range(3):
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
    
    plt.figure("Mean Joint Amplitude vs Frequency", figsize=(10, 6))
    # Use your provided plot_time_histories function.
    # The x-axis is frequency, and the state array has 3 columns (one per DR case).
    plot_time_histories(
        frequencies,
        DR_cases,
        labels=labels,
        title="Mean Joint Amplitude vs Frequency",
        ylabel="Mean Joint Amplitude",
        xlabel="Frequency [Hz]",
        closefig=True,  # Keep the figure open for further modification (like legend)
        savepath=save_dir + savename
    )
    # plt.legend()
    # plt.show()

def question_2_4(muscle_parameters_tag):
    # Ensure that SimulationParameters, run_multiple, load_object, and plot_time_histories are imported or defined

    prepath = './logs/exercise1/question_2_4/'

    frequencies = np.arange(3, 41, 2)  # frequencies from 3 to 40 Hz
    # frequencies = np.arange(3, 6, 2)
    nsim = frequencies.shape[0]
    TWL = np.arange(0.0, 2.1, 0.2)      # TWL values from 0.0 to 2.0
    # TWL = np.arange(0.0, 0.5, 0.2)
    amp = 0.5

    pars_list = [
        SimulationParameters(
            simulation_i=i + j * nsim,
            n_iterations=1001,
            log_path=prepath,
            amp = amp,
            freq=freq,
            twl=twl,
            video_record=False,
            headless=True,
            controller="empty",
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
            cot =controller.metrics["mech_cot"]
            speed_list.append(speed)
            cot_list.append(cot)
        TWL_speed_list.append(speed_list)
        TWL_cot_list.append(cot_list)

    TWL_speeds = np.array(TWL_speed_list).T
    TWL_cot = np.array(TWL_cot_list).T

    labels = [f"TWL = {twl}" for twl in TWL]
    save_dir = './plots/exercise1/question_2_4/'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure("Mechanical speed forward vs Frequency", figsize=(10, 6))
    plot_time_histories(
        frequencies,
        TWL_speeds,
        labels=labels,
        title="Mechanical speed forward vs Frequency",
        ylabel="Mechanical speed forward",
        xlabel="Frequency [Hz]",
        closefig=True,
        savepath = save_dir+muscle_parameters_tag
    )

    plt.figure("Mechanical cot vs Frequency", figsize=(10, 6))
    plot_time_histories(
        frequencies,
        TWL_cot,
        labels=labels,
        title="Mechanical cot vs Frequency",
        ylabel="Mechanical cot",
        xlabel="Frequency [Hz]",
        closefig=True,
        savepath = save_dir+muscle_parameters_tag+"_cot"
    )
    # plt.legend()
    # plt.show()

def exercise1():

    pylog.info("Ex 1")
    pylog.info("Implement exercise 1")
    prepath = './logs/exercise1/'
    
    #The other one to test 
    animal_pose = [0.0, 0.0, 0.1, 0.0, 0, -1.570796327] # Should be 10 meters above ground but error in the pdf
    animal_pose_default = [0.0, 0.0, -0.01, 0.0, 0, -1.570796327]

    # Question 2.2
    question_2_2()
    damping_factor = 1

    # pars = SimulationParameters(
    #         simulation_i=0,
    #         n_iterations=5001,
    #         log_path= prepath,
    #         video_record=False,
    #         compute_metrics='mechanical',
    #         headless=False,
    #         controller="empty",
    #         damping_factor=damping_factor,
    #         amp=0.,
    #         twl=0,
    #         freq=0,
    #         animal_pose = animal_pose,
    #         gravity=np.zeros(3),
    #         joint_poses=0.3*np.ones(15), # joint angles at 0.3pi
    #         print_metrics=False,
    #         return_network=True
    #     )
    # controller = run_single(
    #     pars
    # )
    # print(controller.metrics["mech_speed_fwd"])
    # print(controller.__dict__.keys())
    
    # 
    

    # Question 2.3
    # question_2_3(animal_pose_default, twl=0, savename="water_twl_0")
    # question_2_3(animal_pose, twl=0, savename="space_twl_0")
    # question_2_3(animal_pose_default, twl=0.5, savename="water_twl_05")
    # question_2_3(animal_pose, twl=0.5, savename="space_twl_05")

    # Question 2.4
    muscle_parameters_tags = ["FN_5000_ZC_1000_G0_419",
                              "FN_7500_ZC_1000_G0_419",
                              "FN_10000_ZC_1000_G0_419"]
    # for muscle_parameters_tag in muscle_parameters_tags:
    #     question_2_4(muscle_parameters_tag)
    


if __name__ == '__main__':

    exercise1()

