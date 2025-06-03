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
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros(
            [self.n_oscillators, self.n_oscillators,])
        self.phase_bias              = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates                   = np.zeros(self.n_oscillators)
        self.nominal_amplitudes      = np.zeros(self.n_oscillators)
        self.update(pars)

        # States
        self.n_eq = self.n_oscillators*2  # oscillator phase + oscillator amp
        self.state = np.zeros([self.n_iterations, self.n_eq])
        self.dstate = np.zeros([self.n_eq])  # derivative state

        # State index
        self.oscillator_phase_l = 2 * np.arange(0, self.pars.n_joints)
        self.oscillator_phase_r = 2 * np.arange(0, self.pars.n_joints) + 1
        self.oscillator_phase_all = np.arange(0, 2*self.pars.n_joints)
        self.oscillator_amplitude_l = self.pars.n_joints * 2 + 2 * np.arange(0, self.pars.n_joints)
        self.oscillator_amplitude_r = self.pars.n_joints * 2 + 2 * np.arange(0, self.pars.n_joints) + 1
        self.oscillator_amplitude_all = self.pars.n_joints * \
            2 + np.arange(0, 2*self.pars.n_joints)

        # Initial state
        if self.pars.initial_phases is None:
            self.state[0, 0:2*self.pars.n_joints-1:2] = 1 * \
                np.linspace(2*np.pi, 0, self.pars.n_joints)
            self.state[0, 1:2*self.pars.n_joints:2] = np.linspace(np.pi, -np.pi, self.pars.n_joints)
        else:
            self.state[0, :2*self.pars.n_joints] = self.pars.initial_phases

        self.state[0, self.n_oscillators:2 * self.n_oscillators] = np.zeros(self.n_oscillators)

        # motor output and indexes
        self.motor_out = np.zeros([self.n_iterations, self.n_oscillators])
        self.motor_l = 2*np.arange(0, self.pars.n_joints)
        self.motor_r = self.motor_l + 1

        # initialize ode solver
        self.f = self.network_ode
        self.step = self.step_euler

        # pre-computed zero activity for the last two tail joints
        self.zeros4 = np.zeros(4)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        self.freqs[:self.n_oscillators] = 2*np.pi*(parameters.cpg_frequency_gain*parameters.drive + parameters.cpg_frequency_offset)

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if abs(i-j)==2:
                    self.coupling_weights[i, j] =  parameters.weights_body2body
                elif j-i==1 and i%2==0:
                    self.coupling_weights[i, j] =  parameters.weights_body2body_contralateral
                elif i-j==1 and i%2==1:
                    self.coupling_weights[i, j] =  parameters.weights_body2body_contralateral
                else:
                    self.coupling_weights[i, j] = 0


    def set_phase_bias(self, parameters):
        """Set phase bias"""
        parameters.phase_lag_body /= (parameters.n_joints+parameters.n_passive_joints-1)
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if abs(i-j)==2:
                    self.phase_bias[i, j] = np.sign(i-j) * parameters.phase_lag_body
                elif j-i==1 and i%2==0:
                    self.phase_bias[i, j] = np.sign(i-j) * np.pi
                elif i-j==1 and i%2==1:
                    self.phase_bias[i, j] = np.sign(i-j) * np.pi
                else:
                    self.phase_bias[i, j] = 0

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        self.nominal_amplitudes[0:2*parameters.n_joints-1:2] = parameters.drive * parameters.cpg_amplitude_gain
        self.nominal_amplitudes[1:2*parameters.n_joints:2] = parameters.drive * parameters.cpg_amplitude_gain

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates[:] = parameters.amplitude_rates

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
        """
        n_oscillators = self.n_oscillators

        phases = state[:n_oscillators]
        amplitudes = state[n_oscillators:2*n_oscillators]
        phase_repeat = np.repeat(
            np.expand_dims(phases, axis=1),
            n_oscillators,
            axis=1,
        )

        dphases = (
            # Intrinsic frequencies
            self.freqs
            # Coupling
            + np.sum(
                amplitudes * self.coupling_weights.T * np.sin(
                    phase_repeat.T - phase_repeat + self.phase_bias.T),
                axis=1,
            )
        )

        damplitudes = (
            # Amplitude dynamics
            self.rates * ( self.nominal_amplitudes - amplitudes)
        )

        # hyper = define_hyperparameters()
        # ws_ref = hyper["ws_ref"]

    
        # -----Entraining stretch feedback-----
        stretch = np.zeros(n_oscillators)
        if pos is not None:
            # Ipsilateral stretch feedback
            #W_ipsi = ws_ref * self.pars.feedback_weights_ipsi
            W_ipsi   = self.pars.feedback_weights_ipsi
            # Contralateral stretch feedback
            #W_contra = ws_ref * self.pars.feedback_weights_contra
            W_contra = self.pars.feedback_weights_contra

            # Compute stretch feedback for each oscillator
            for seg in range(self.pars.n_joints):
                alpha = pos[seg]    # alpha seg

                # LEFT side
                stretch[2*seg] = W_ipsi * max(0, +alpha) + W_contra * max(0, -alpha)

                # RIGHT side
                stretch[2*seg + 1] = W_ipsi * max(0, -alpha) + W_contra * max(0, +alpha)

        
        # -----Amplitude derivative-----
        for i in range(n_oscillators):
            
            if self.nominal_amplitudes[i] > 1e-6: # For numerical stability
                dphases[i] -= (stretch[i] / self.nominal_amplitudes[i]) * np.sin(phases[i])
            
            damplitudes[i] +=  stretch[i] * np.cos(phases[i])

        return np.concatenate([dphases, damplitudes])

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
        """
        phase = self.state[iteration, self.oscillator_phase_all]
        amplitude = self.state[iteration, self.oscillator_amplitude_all]
        oscillator_output = amplitude*(1+np.cos(phase))
        motor_output = self.pars.motor_output_scaling*oscillator_output

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
        """

        self.state[iteration+1, :] = (
            self.state[iteration, :] +
            timestep * self.f(self.state[iteration], pos=pos)
        )

        return np.concatenate([
            self.motor_output(iteration),  # the active joints
            self.zeros4  # the last (tail) passive joint
        ])
