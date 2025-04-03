
from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import matplotlib.pyplot as plt
from plotting_common import plot_time_histories, plot_time_histories_multiple_windows
import numpy as np
from util.rw import load_object

num_process = 10

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
    0.14929      # Note: Tail moves passively,
])  # type: ignore unit:radian

def question_3_3():

    log_path = './logs/exercise3/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations=5001,
        controller="abstract oscillator",
        log_path=log_path,
        compute_metrics=None,
        print_metrics=False,
        return_network=True,
        drive = 1,
        cpg_frequency_gain = 0.6,
        cpg_frequency_offset = 0.6,
        cpg_amplitude_gain = 0.125 * np.ones(13),
        weights_body2body = 30,
        weights_body2body_contralateral = 10,
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

    # print(controller.__dict__.keys())
    # print(controller.oscillator_phase_all.shape)
    phases = controller.state[:, :2*all_pars.n_joints]
    # phases = (phases + np.pi) % (2*np.pi) - np.pi  # constrain phases to [-pi, pi]

    amplitudes = controller.state[:, 2*all_pars.n_joints:]
    motor_l = controller.motor_out[:, controller.motor_l]
    motor_r = controller.motor_out[:, controller.motor_r]
    motor_diff = motor_l - motor_r
    joint_angles = controller.joints_positions # divide by pi for real angles?
    # joints_positions = controller.joints_positions
    print(phases.shape)
    print(amplitudes.shape)
    print(motor_l.shape)
    print(motor_r.shape)
    print(motor_diff.shape)
    print(joint_angles.shape)

    # state = np.concatenate([phases, amplitudes, motor_l, motor_r, motor_diff, joint_angles], axis=1)
    plt.figure("phases")
    plot_time_histories(controller.times, phases, ylabel=None, closefig= True, savepath = save_dir+"phases")

    plt.figure("amplitudes")
    plot_time_histories(controller.times, amplitudes, ylabel=None, closefig= True, savepath = save_dir+"amplitudes")
    
    plt.figure("motor_l")
    plot_time_histories(controller.times, motor_l, ylabel=None, closefig= True, savepath = save_dir+"motor_l")

    plt.figure("motor_r")
    plot_time_histories(controller.times, motor_r, ylabel=None, closefig= True, savepath = save_dir+"motor_r")

    plt.figure("motor_diff")
    plot_time_histories(controller.times, motor_diff, ylabel=None, closefig= True, savepath = save_dir+"motor_diff")

    plt.figure("joint_angles")
    plot_time_histories(controller.times, joint_angles, ylabel=None, closefig= True, savepath = save_dir+"joint_angles")
    
    plt.show()

def question_3_4():

    # Also test with the simulation

    drives = np.linspace(1, 9, 8).astype(np.int8)
    prepath = './logs/exercise3/question_3_4/'
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=5001,
            log_path= prepath,
            video_record=False,
            headless=True,
            controller="abstract oscillator",
            print_metrics=False,
            compute_metrics='mechanical',
            drive = drive,
        )
        for i, drive in enumerate(drives)
    ]

    run_multiple(pars_list, num_process=num_process)
    save_dir = './plots/exercise3/question_3_4/'
    os.makedirs(save_dir, exist_ok=True)


    cot_list = []
    speed_list = []
    mech_mean_freq_list = []
    mech_mean_amp_list = []

    for i in range(len(drives)):
        # load controller
        controller = load_object(prepath+"controller"+str(i))
        motor_output = controller.motor_out
        n_joints = motor_output.shape[1]
        motor_labels = [f"motor_out {i}" for i in range(n_joints)]

        # plt.figure("motor output")
        plot_time_histories(controller.times, 
                            motor_output, 
                            labels=motor_labels, 
                            title="Motor outputs", 
                            closefig=True, 
                            savepath=save_dir + "drive_"+str(drives[i])#.replace('.', '_')
                            )
        
        cot = controller.metrics["mech_cot"]
        speed = controller.metrics["mech_speed_fwd"]
        mech_mean_freq = controller.metrics["mech_mean_frequency"]
        mech_mean_amp = controller.metrics["mech_mean_amplitude"]

        cot_list.append(cot)
        speed_list.append(speed)
        mech_mean_freq_list.append(mech_mean_freq)
        mech_mean_amp_list.append(mech_mean_amp)

    cot = np.array(cot_list)
    speed = np.array(speed_list)
    mech_mean_freq = np.array(mech_mean_freq_list)
    mech_mean_amp = np.array(mech_mean_amp_list)

    states = np.column_stack((cot, speed, mech_mean_freq, mech_mean_amp))
    print(states.shape)
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
        drive = 4,
        cpg_frequency_gain = 0.6,
        cpg_frequency_offset = 0.6,
        cpg_amplitude_gain = 0.125 * np.ones(13),
        weights_body2body = 30,
        weights_body2body_contralateral = 10,
        phase_lag_body = 2*np.pi,
        amplitude_rates = 1,
        motor_output_scaling = 1,
    )

    pylog.info("Running the simulation")
    # controller = run_single(
    #     all_pars
    # )

    question_3_3()
    # question_3_4()
    
    # Hint: Optionally you can use some helper function to generate the plots
    # such as (plot_time_histories)


if __name__ == '__main__':

    exercise3()

