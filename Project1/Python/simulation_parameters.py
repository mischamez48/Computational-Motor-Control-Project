
"""Simulation parameters"""
import numpy as np
from farms_core import pylog


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        """
        Class containing all neuromechanical model parameters
        Inputs:
        kwargs: extra parameter arguments (these override previous declarations)
        """
        super(SimulationParameters, self).__init__()
        # simulation parameters
        self.n_joints = 13  # number of active joints
        self.n_total_joints = 15  # number of total joints (active+passive)
        self.n_passive_joints = 2  # number of passive joints
        self.timestep = 0.001  # integration time step
        self.n_iterations = 4001  # number of integration time steps
        self.timestep = 0.001  # integration time step
        self.n_iterations = 4001  # number of integration time steps

        # gui/recording parameters
        self.headless = True  # For headless mode (No GUI, could be faster)
        self.fast = False  # For fast mode (not real-time)
        self.video_record = False  # For saving the video
        self.video_speed = 1.0  # video speed
        # path where the simulation data will be stored (no simulation data
        # will be saved if the string is empty)
        self.log_path = ""
        # video name (saved in the log_path folder under the video_name name)
        self.video_name = "video"
        self.video_fps = 50  # frames per second
        self.camera_id = 0  # camera type: 0=angles top view, 1=top view, 2=side view, 3=back view
        self.show_progress = True  # show progress bar of running the simulation
        # simulation id (of log_path!="" saves the simulation in the log_path
        # folder under the name "simulation_i")
        self.simulation_i = 0
        # None = no metrics, 'neural' = neural metrics, 'mechanical' =
        # mechanical metrics, 'all' = all metrics (metrics are stored in
        # network.metrics)
        self.compute_metrics = 'all'
        self.print_metrics = True  # if True print all computed metrics
        # if True, run_single_sim will return the controller class
        self.return_network = False
        # (keep it False when running simulations on mutliple cpu cores,
        # or it will saturate the RAM memory)
        self.random_spine = False  # if True, initialize the joints angles in a random position

        self.gravity = np.array([0, 0, -9.81])  # gravity vector
        self.damping_factor = 1.0  # damping factor for the joints
        # animal initial pose (x, y, z, roll, pitch, yaw)
        self.animal_pose = [0.0, 0.0, -0.01, 0.0, 0, -1.570796327]
        self.joint_poses = np.zeros(
            self.n_total_joints)  # initial joint angles

        self.muscle_parameters_tag = 'FN_6000_ZC_1000_G0_419'  # muscle param file
        # "sine" for using the WaveController, "abstract oscillator" for abstract oscillator
        self.controller = 'empty'

        # default parameters
        self.method = "euler"  # integration method (euler or noise)
        self.initial_phases = None  # np.linspace(2*np.pi, 0, 30)

        # parameters of the sine controller

        # parameters of the abstract oscillator
        self.drive = 4  # drive to the abstract oscillator controller
        self.cpg_frequency_gain = 0.6
        self.cpg_frequency_offset = 0.6
        self.cpg_amplitude_gain = 0.125 * np.ones(self.n_joints)
        # Coupling weigths between neighborhood segments
        self.weights_body2body = 30
        # Coupling weigths between contralateral segments
        self.weights_body2body_contralateral = 10
        self.phase_lag_body = 2*np.pi  # Total phase lag from body to tail along the spine
        self.amplitude_rates = 20  # Convergence rates of amplitudes

        self.motor_output_scaling = 1  # muscle force scaling factor G

        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

