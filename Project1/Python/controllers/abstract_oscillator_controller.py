"""Oscillator network ODE"""

import numpy as np
import scipy.stats as ss
from farms_core import pylog


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
        self.oscillator_phase_l = np.arange(0, self.pars.n_joints)
        self.oscillator_phase_r = self.pars.n_joints + \
            np.arange(0, self.pars.n_joints)
        self.oscillator_phase_all = np.arange(0, 2*self.pars.n_joints)
        self.oscillator_amplitude_l = self.pars.n_joints * \
            2 + np.arange(0, self.pars.n_joints)
        self.oscillator_amplitude_r = self.pars.n_joints * \
            3 + np.arange(0, self.pars.n_joints)
        self.oscillator_amplitude_all = self.pars.n_joints * \
            2 + np.arange(0, 2*self.pars.n_joints)

        # Initial state
        if self.pars.initial_phases is None:
            self.state[0, :self.pars.n_joints] = 1 * \
                np.linspace(2*np.pi, 0, self.pars.n_joints)
            self.state[0, self.pars.n_joints:2 *
                       self.pars.n_joints] = np.linspace(np.pi, -
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

    def network_ode(self, state):
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
        dphases = np.zeros(n_oscillators)
        damplitudes = np.zeros(n_oscillators)
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
        motor_output = np.zeros(self.n_oscillators)
        self.motor_out[iteration, :] = motor_output

        return motor_output

    def step_euler(self, iteration, timestep):
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
        self.state[iteration+1, :] = self.state[iteration, :]
        return (np.zeros(30))

