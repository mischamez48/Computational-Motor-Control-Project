
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import matplotlib.pyplot as plt
from plotting_common import plot_time_histories, plot_time_histories_multiple_windows
import numpy as np

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
        drive = 4,
        cpg_frequency_gain = 0.6,
        cpg_frequency_offset = 0.6,
        cpg_amplitude_gain = 0.125 * np.ones(13),
        weights_body2body = 30,
        weights_body2body_contralateral = 10,
        phase_lag_body = 2*np.pi,
        amplitude_rates = 20,
        motor_output_scaling = 1,
        # headless=False,
    )

    controller = run_single(
        all_pars
    )

    save_dir = './plots/exercise3/question_3_3/'
    os.makedirs(save_dir, exist_ok=True)

    print(controller.__dict__.keys())
    # print(controller.oscillator_phase_all.shape)
    phases = controller.state[:, :2*all_pars.n_joints]
    amplitudes = controller.state[:, 2*all_pars.n_joints:]
    motor_l = controller.motor_out[:, controller.motor_l]
    motor_r = controller.motor_out[:, controller.motor_r]
    motor_diff = motor_l - motor_r
    joint_angles = controller.joints_positions # divide by pi for real angles?
    # joints_positions = controller.joints_positions 

    # state = np.concatenate([phases, amplitudes, motor_l, motor_r, motor_diff, joint_angles], axis=1)
    print(phases[:,0].max())
    print(amplitudes[:, 0].max())
    plot_time_histories(controller.times, phases, closefig= False)
    # plot_time_histories(controller.times, motor_l, closefig= False)
    # plot_time_histories(controller.times, motor_r, closefig= False)
    # plot_time_histories(controller.times, motor_diff, closefig= False)
    plt.show()

def exercise3():

    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")
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
        cpg_amplitude_gain = 0.5 * np.ones(13),
        weights_body2body = 30,
        weights_body2body_contralateral = 10,
        phase_lag_body = 2*np.pi,
        amplitude_rates = 20,
        motor_output_scaling = 1,
    )

    pylog.info("Running the simulation")
    # controller = run_single(
    #     all_pars
    # )
    question_3_3()
    
    # Hint: Optionally you can use some helper function to generate the plots
    # such as (plot_time_histories)


if __name__ == '__main__':

    exercise3()

