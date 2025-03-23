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
        # Get current time
        current_time = iteration * timestep

        # Initialize motor activations array
        motor_activations = np.zeros(2 * self.n_total_joints)

        # Get parameters from self.pars
        frequency = self.pars.freq
        amplitude = self.pars.amp
        total_wave_lag = self.pars.twl

        # Loop through each joint to calculate its activation
        for i in range(self.n_total_joints):
           # Calculate position index (normalized by the number of total joints)
            pos_index = i / (self.n_joints)

            # Calculate left muscle activation using the equation from the instructions
            ml = 0.5 + amplitude/2 * np.sin(2*np.pi * (frequency * current_time - 
                                                    total_wave_lag * pos_index))
            
            # Calculate right muscle activation using the equation from the instructions
            mr = 0.5 - amplitude/2 * np.sin(2*np.pi * (frequency * current_time - 
                                                    total_wave_lag * pos_index))
            
            # Set the left and right muscle activations in the array
            motor_activations[2*i] = ml     # Left muscle (even indexes)
            motor_activations[2*i+1] = mr   # Right muscle (odd indexes)



        # Store the motor activations for later use
        self.motor_out[iteration, :] = motor_activations
        
        return motor_activations

