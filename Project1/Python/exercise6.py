import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt

from util.rw import load_object
from util.run_closed_loop import run_multiple, run_single
from simulation_parameters import SimulationParameters
from plotting_common import plot_1d, save_figures, plot_2d, plot_time_histories_multiple_windows, plot_time_histories

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

NUM_PROCESS = 10

def question_5_4():
    """
    Test different feedback weight values for ipsilateral and contralateral feedback connections. 
    Keep w_ipsi = 0 and test w_contra in range [-1,1] scaled by FEEDBACK_GAIN_REF. Keep w_contra = 0 
    and test w_ipsi in range [-1,1] scaled by FEEDBACK_GAIN_REF. FEEDBACK_GAIN_REF is the inverse of 
    average joint amplitudes (provided in the code). 
    
    Plot the neural controller (neural frequency, total wave lag ) and mechanical metrics (cost of 
    transport, energy consumption, forward speed, joint amplitudes, sum of torques) as a function of 
    different ipsilateral and contralateral feedback strengths..
    """
    no_ipsi_simulations()
    no_contra_simulations()

def plt_exercise5_4(w, prepath, case="no_ipsi"):
    """
    Plot neural and mechanical metrics as a function of contralateral feedback strength.
    Joint amplitudes are plotted per joint.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    save_dir = f'./plots/exercise6/question_5_4/{case}/'
    os.makedirs(save_dir, exist_ok=True)

    # Lists for scalar metrics
    neural_freq_list = []
    total_wave_lag_list = []
    cot_list = []
    energy_list = []
    speed_list = []
    sum_torque_list = []
    joint_amp_matrix = []

    # Load metrics
    for i in range(len(w)):
        controller = load_object(prepath + f"controller{i}")

        neural_freq = controller.metrics["neur_frequency"]
        total_wave_lag = controller.metrics["neur_twl"]
        cot = controller.metrics["mech_cot"]
        energy = controller.metrics["mech_energy"]
        speed = controller.metrics["mech_speed_fwd"]
        joint_amp = controller.metrics["mech_joint_amplitudes"]
        sum_torque = controller.metrics["mech_torque"]

        w_contra = w[i]

        neural_freq_list.append([w_contra, neural_freq])
        total_wave_lag_list.append([w_contra, total_wave_lag])
        cot_list.append([w_contra, cot])
        energy_list.append([w_contra, energy])
        speed_list.append([w_contra, speed])
        sum_torque_list.append([w_contra, sum_torque])
        joint_amp_matrix.append(joint_amp)

    # Convert scalar metric lists to arrays
    neural_freq = np.array(neural_freq_list)
    total_wave_lag = np.array(total_wave_lag_list)
    cot = np.array(cot_list)
    energy = np.array(energy_list)
    speed = np.array(speed_list)
    sum_torque = np.array(sum_torque_list)
    joint_amp_matrix = np.array(joint_amp_matrix)  # shape: (n_conditions, n_joints)


    if case == "no_ipsi":
        weight_name = "w_contra"
    else:
        weight_name = "w_ipsi"
    # Plot 1D scalar metrics
    metric_configs = [
        (neural_freq, [weight_name, "Neural Frequency [Hz]"], "neural_frequency"),
        (total_wave_lag, [weight_name, "Total Wave Lag [rad]"], "total_wave_lag"),
        (cot, [weight_name, "Cost of Transport"], "cost_of_transport"),
        (energy, [weight_name, "Energy [J]"], "energy"),
        (speed, [weight_name, "Forward Speed [m/s]"], "forward_speed"),
        (sum_torque, [weight_name, "Sum of Torques [Nm]"], "sum_torque"),
    ]

    for data, labels, filename in metric_configs:
        plt.figure(filename)
        plot_1d(data, labels)
        plt.title(labels[1])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename}.png"))
        plt.close()

    # Plot joint amplitudes across conditions
    plt.figure("joint_amplitudes_per_joint")
    n_joints = joint_amp_matrix.shape[1]
    for j in range(n_joints):
        plt.plot(w, joint_amp_matrix[:, j], label=f'Joint {j+1}')
    plt.xlabel(weight_name)
    plt.ylabel("Joint Amplitude [rad]")
    plt.title("Joint Amplitudes per Feedback Strength")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "joint_amplitudes_per_joint.png"))
    plt.close()

    print(f"Saved plots to: {save_dir}")

def no_ipsi_simulations():

    w_contra_range = np.linspace(-1, 1, 20)
    prepath = './logs/exercise6/question_5_4/no_ipsi/'
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=5001,
            log_path= prepath,
            video_record=False,
            headless=True,
            controller="abstract oscillator",
            print_metrics=False,
            compute_metrics='all',
            cpg_amplitude_gain = REF_JOINT_AMP[:-2], #
            feedback_weights_ipsi = 0.0,
            feedback_weights_contra = w_contra,
        )
        for i, w_contra in enumerate(w_contra_range)
    ]

    run_multiple(pars_list, num_process=NUM_PROCESS)

    plt_exercise5_4(w_contra_range, prepath, case = "no_ipsi" )

def no_contra_simulations():

    w_ipsi_range = np.linspace(-1, 1, 20)
    prepath = './logs/exercise6/question_5_4/no_contra/'
    pars_list = [
        SimulationParameters(
            simulation_i=i,
            n_iterations=5001,
            log_path= prepath,
            video_record=False,
            headless=True,
            controller="abstract oscillator",
            print_metrics=False,
            compute_metrics='all',
            cpg_amplitude_gain = REF_JOINT_AMP[:-2], #[:-2]
            feedback_weights_ipsi =  w_ipsi,
            feedback_weights_contra = 0.0,
        )
        for i, w_ipsi in enumerate(w_ipsi_range)
    ]

    run_multiple(pars_list, num_process=NUM_PROCESS)
    w_scaled = w_ipsi_range
    plt_exercise5_4(w_scaled, prepath, case = "no_contra" )

def question_5_4_2d():
    """
    Sweep w_ipsi and w_contra values in [-1, 1] scaled by FEEDBACK_GAIN_REF.
    Plot all neural and mechanical metrics over the 2D grid.
    """
    run_2d_feedback_sweep()

def run_2d_feedback_sweep():
    w_range = np.linspace(-1, 1, 20)
    prepath = './logs/exercise6/question_5_4/full_2d/'
    pars_list = []

    for i, w_ipsi in enumerate(w_range):
        for j, w_contra in enumerate(w_range):
            sim_index = i * len(w_range) + j
            pars_list.append(SimulationParameters(
                simulation_i=sim_index,
                n_iterations=5001,
                log_path=prepath,
                video_record=False,
                headless=True,
                controller="abstract oscillator",
                print_metrics=False,
                compute_metrics='all',
                cpg_amplitude_gain=REF_JOINT_AMP[:-2], 
                feedback_weights_ipsi=w_ipsi,
                feedback_weights_contra=w_contra,
            ))

    run_multiple(pars_list, num_process=NUM_PROCESS)

    # Store the parameter pairs used
    param_grid = [(w_ipsi, w_contra) for w_ipsi in w_range for w_contra in w_range]
    plt_exercise5_4_2d(param_grid, prepath)

def plt_exercise5_4_2d(param_grid, prepath):
    """
    Plot all metrics using plot_2d over w_ipsi vs w_contra.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    save_dir = './plots/exercise6/question_5_4/full_2d/'
    os.makedirs(save_dir, exist_ok=True)

    # Data storage: each is (w_ipsi, w_contra, metric_value)
    neural_freq = []
    total_wave_lag = []
    cot = []
    energy = []
    speed = []
    sum_torque = []

    for i, (w_ipsi, w_contra) in enumerate(param_grid):
        controller = load_object(prepath + f"controller{i}")

        neural_freq.append([w_ipsi, w_contra, controller.metrics["neur_frequency"]])
        total_wave_lag.append([w_ipsi, w_contra, controller.metrics["neur_twl"]])
        cot.append([w_ipsi, w_contra, controller.metrics["mech_cot"]])
        energy.append([w_ipsi, w_contra, controller.metrics["mech_energy"]])
        speed.append([w_ipsi, w_contra, controller.metrics["mech_speed_fwd"]])
        sum_torque.append([w_ipsi, w_contra, controller.metrics["mech_torque"]])

    metric_data = [
        (np.array(neural_freq), ["w_ipsi", "w_contra", "Neural Frequency [Hz]"], "neural_frequency"),
        (np.array(total_wave_lag), ["w_ipsi", "w_contra", "Total Wave Lag [rad]"], "total_wave_lag"),
        (np.array(cot), ["w_ipsi", "w_contra", "Cost of Transport"], "cost_of_transport"),
        (np.array(energy), ["w_ipsi", "w_contra", "Energy [J]"], "energy"),
        (np.array(speed), ["w_ipsi", "w_contra", "Forward Speed [m/s]"], "forward_speed"),
        (np.array(sum_torque), ["w_ipsi", "w_contra", "Sum of Torques [Nm]"], "sum_torque"),
    ]

    for data, labels, filename in metric_data:
        plt.figure(filename)
        plot_2d(data, labels, cmap='viridis')
        plt.title(labels[2])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename}.png"))
        plt.close()

    print(f"2D plots saved to {save_dir}")

def exercise6():

    question_5_4()
    question_5_4_2d()


if __name__ == '__main__':

    exercise6()
