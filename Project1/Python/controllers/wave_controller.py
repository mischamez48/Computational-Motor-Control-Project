"""Network controller"""

import numpy as np
import farms_pylog as pylog


class WaveController:

    """Test controller"""

    def __init__(self, pars):
        self.pars = pars
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            pars.n_iterations *
            pars.timestep,
            pars.n_iterations)
        self.n_total_joints = pars.n_total_joints
        self.n_joints = pars.n_joints
        self.n_oscillators = 2 * (self.n_total_joints)

        # motor output array for recording the motor outputs
        self.motor_out = np.zeros((pars.n_iterations, self.n_oscillators))
        pylog.warning(
            "Implement below the step function following the instructions here and in the report")

        # indexes of the left muscle activations (optional)
        self.motor_l = 2*np.arange(self.n_total_joints)
        # indexes of the right muscle activations (optional)
        self.motor_r = self.motor_l + 1

    def step(self, iteration, timestep, pos=None):
        """
        Step function. This function passes the activation functions of the muscle model
        Inputs:
        - iteration - iteration index
        - time - time vector
        - timestep - integration timestep
        - pos (not used) - joint angle positions

        Implement here the control step function,
        it should return an array of 2*n_joint=30 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations

        In addition to returning the activation functions, store
        them in self.motor_out for later use offline
        """
        return np.zeros(30)

