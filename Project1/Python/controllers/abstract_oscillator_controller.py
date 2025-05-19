"""Oscillator network ODE"""

import numpy as np
import scipy.stats as ss
from farms_core import pylog
from util.zebrafish_hyperparameters import define_hyperparameters


class AbstractOscillatorController:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

        # Simulation parameters
        self.pars = pars
        self.n_iterations = pars.n_iterations
        self.timestep = pars.timestep
        self.times = np.linspace(
            0, self.pars.n_iterations*self.timestep, self.pars.n_iterations)

        # Abstract oscillator parameters
        self.n_oscillators = 2*self.pars.n_joints

        # States
        self.n_eq = self.n_oscillators*2  # oscillator phase + oscillator amp
        self.state = np.zeros([self.n_iterations, self.n_eq])
        self.dstate = np.zeros([self.n_eq])  # derivative state

        # State index
        self.oscillator_phase_l = 2 * np.arange(0, self.pars.n_joints)
        self.oscillator_phase_r = 2 * np.arange(0, self.pars.n_joints) + 1
        self.oscillator_phase_all = np.arange(0, 2*self.pars.n_joints)
        self.oscillator_amplitude_l = self.pars.n_joints * \
            2 + 2 * np.arange(0, self.pars.n_joints)
        self.oscillator_amplitude_r = self.pars.n_joints * \
            2 + 2 * np.arange(0, self.pars.n_joints) + 1
        self.oscillator_amplitude_all = self.pars.n_joints * \
            2 + np.arange(0, 2*self.pars.n_joints)

        # Initial state
        if self.pars.initial_phases is None:
            self.state[0, 0:2*self.pars.n_joints-1:2] = 1 * \
                np.linspace(2*np.pi, 0, self.pars.n_joints)
            self.state[0, 1:2 *
                       self.pars.n_joints:2] = np.linspace(np.pi, -
                                                           np.pi, self.pars.n_joints)
        else:
            self.state[0, :2*self.pars.n_joints] = self.pars.initial_phases

        self.state[0, self.n_oscillators:2 *
                   self.n_oscillators] = np.zeros(self.n_oscillators)

        # motor output and indexes
        self.motor_out = np.zeros([self.n_iterations, self.n_oscillators])
        self.motor_l = 2*np.arange(0, self.pars.n_joints)
        self.motor_r = self.motor_l + 1

        # initialize ode solver
        self.f = self.network_ode
        self.step = self.step_euler

        # pre-computed zero activity for the last two tail joints
        self.zeros4 = np.zeros(4)

    def network_ode(self, state, pos=None):
        """
        pars
        -------
        self: AbstractOscillatorController
            The controller object
        state: <np.array>
            An array of size 2*n_oscillators storing the oscillator phases and amplitudes
        Returns
        -------
        dstate: <np.array>
            An array of size 2*n_oscillators storing the oscillator phases and amplitudes derivatives
            Note that phases and amplitudes have to appear in the same order as defined in states.
        -------
        This function is called each step to update the network states (amplitudes and phases).
        Here you have to implement the Ordinary Differential Equation (ODE)
        to compute the derivatives of network states.
        For which you need CPG parameters  like nominal amplitudes, coupling weights, rates.
        The computation of the above-mentioned parameters can go in another custom function or
        be implemented here directly.
        """
        n_oscillators = self.n_oscillators
        # Implement equation here

        # -----Init-----
        hyper = define_hyperparameters()
        ws_ref = hyper["ws_ref"]

        # -----Needed parameters-----
        theta = state[self.oscillator_phase_all]
        # f= Gfreq·d+ offset cf. project 1
        frequency = self.pars.cpg_frequency_gain * self.pars.drive + self.pars.cpg_frequency_offset
        # coupling weights
        weight_body = self.pars.weights_body2body
        weight_contra = self.pars.weights_body2body_contralateral
        # phase lag
        phi_total = self.pars.phase_lag_body
        phi_body_total = phi_total / ( self.pars.n_joints - 1)
        # phase derivative
        dtheta = np.zeros(n_oscillators)
        # amplitude
        r = state[self.oscillator_amplitude_all]
        # amplitude derivative
        dr = np.zeros(n_oscillators)
        # amplitude convergence rate
        a = self.pars.amplitude_rates
        # amplitude gain
        R = self.pars.cpg_amplitude_gain
    
        # -----Entraining stretch feedback-----
        stretch = np.zeros(n_oscillators)
        if pos is not None:
            # Ipsilateral stretch feedback
            W_ipsi = ws_ref * self.pars.feedback_weights_ipsi
            # Contralateral stretch feedback
            W_contra = ws_ref * self.pars.feedback_weights_contra

            # Compute stretch feedback for each oscillator
            for seg in range(self.pars.n_joints):
                alpha = pos[seg]    # alpha seg

                # LEFT side
                stretch[2*seg] = W_ipsi * max(0, +alpha) + W_contra * max(0, -alpha)

                # RIGHT side
                stretch[2*seg + 1] = W_ipsi * max(0, -alpha) + W_contra * max(0, +alpha)


        # -----Phase derivative-----
        for i in range(n_oscillators):
            # 2 pi f component in the formula 6
            dtheta[i] = 2 * np.pi * frequency

            for j in range(n_oscillators):
                # coupling from other oscillators than itself
                if i == j:
                    continue
                
                # coupling logic
                delta = abs(i - j)
                same_side = (i % 2 == j % 2)

                if delta == 2 and same_side:
                    # body2body coupling
                    w_ij = weight_body
                    phi_ij = np.sign(i - j) * phi_body_total

                elif delta == 1 and (
                    (j - i == 1 and i % 2 == 0) or  # left to right
                    (i - j == 1 and i % 2 == 1)     # right to left
                ):
                    w_ij = weight_contra
                    phi_ij = np.sign(i - j) * np.pi
                else:
                    # no coupling
                    continue
                
                # compute the interaction term of the derivative
                dtheta[i] += r[j] * w_ij * np.sin(theta[j] - theta[i] - phi_ij)

            # -----Stretch feedback-----
            # amplitude gain for the oscillator
            
            ####################
            # R_i = R[i // 2]
            # if stretch[i] > 0:
            #     dtheta[i] -= (stretch[i] / R_i) * np.sin(theta[i])
            ####################
            if r[i] > 1e-4: # For numerical stability
                dtheta[i] -= (stretch[i] / r[i]) * np.sin(theta[i])

        
        # -----Amplitude derivative-----
        for i in range(n_oscillators):
            dr[i] = a * (R[i // 2] - r[i]) + stretch[i] * np.cos(theta[i])

        return np.concatenate([dtheta, dr])

    def motor_output(self, iteration):
        """
        pars
        -------
        self: AbstractOscillatorController
            The controller object
            Hint: you can call self.state to access the current state of the controller
        iteration: <int>
            Current sim itertaion
        Returns
        -------
        motor_output: <np.array>
            An array of size 2*n_active_joints storing the muscle activations
        -------
        Here you have to use phase, amplitude and muscle strength to
        finalize muscle activations for the first 13 active joints.
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations

        In addition to returning the motor output, store
        them in self.motor_out for later use offline
        Note: You only update and store the motor output at current iteration.
        i.e. set only self.motor_out[iteration,:]
        """
        phase = self.state[iteration, self.oscillator_phase_all]
        amplitude = self.state[iteration, self.oscillator_amplitude_all]
        oscillator_output = amplitude*(1+np.cos(phase))
        motor_output = self.pars.motor_output_scaling*oscillator_output

        # remapping oscillator to convention
        # motor_output = np.zeros(self.n_oscillators)
        # motor_output[::2] = output[:self.pars.n_joints]
        # motor_output[1::2] = output[self.pars.n_joints:]

        # store muscle output
        self.motor_out[iteration, :] = motor_output
        return motor_output

    def step_euler(self, iteration, timestep, pos=None):
        """
        pars
        -------
        self: AbstractOscillatorController
            The controller object
            Hint: you can call self.state to access the current state of the controller
                  you can call self.f(self, state) to call the network_ode(self, state) function
        iteration: <int>
            Current sim itertaion
        Returns
        -------
        motor_output_all: <np.array>
            An array of size 2*n_joints_total storing the muscle activations for the active and
            passive joints at current sim itertaion
        -------
        Here you have to perform the Euler step on the oscillator states.
        You return the muscle activation of all body joints at current iteration (array of 2*n_joints_total)
        which includes updated motor outputs from active joints and the motor outputs for passive joints.
        """

        self.state[iteration+1, :] = (
            self.state[iteration, :] +
            timestep * self.f(self.state[iteration], pos=pos)
        )

        return np.concatenate([
            self.motor_output(iteration),  # the active joints
            self.zeros4  # the last (tail) passive joint
        ])