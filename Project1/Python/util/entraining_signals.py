"""
Author: Alessandro Pazzaglia
Date: 2023-10-05

This module contains functions to generate and plot poaition signals for multiple chains of degrees of freedom (DOFs).
The main functions include:

- _plot_signals: Plots the generated signals for each chain.
- _position_signal_single_chain: Generates poaition for a single chain of DOFs.
- _position_signal_multi_chain: Generates poaition for multiple chains of DOFs.
- _get_position_signal: Retrieves the poaition signals based on the provided parameters.
- run_example: Runs an example to demonstrate the functionality of the module.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm

from typing import Any, Callable


def _plot_signals(
    times: np.ndarray,
    signals_chains: list[np.ndarray],
    names_chains: list[str],
):
    ''' Plot signals '''

    n_chains = len(signals_chains)

    fig, axs = plt.subplots(
        n_chains,
        1,
        sharex=True,
        figsize=(8.0, 10.5),
        dpi=300
    )
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    axs[0].set_title('Position signals')
    axs[0].set_xlim(times[0], times[-1])
    axs[-1].set_xlabel('Time')
    fig.subplots_adjust(hspace=0)

    for chain, signals in enumerate(signals_chains):

        max_val = np.amax(signals)
        min_val = np.amin(signals)
        range_val = max_val - min_val

        n_signals = signals.shape[1]
        n_joints = n_signals
        signal_jump = range_val * 1.0

        colors = matplotlib.cm.winter(np.linspace(0.10, 0.90, n_joints))

        for joint in range(n_joints):
            axs[chain].plot(
                times,
                + signals[:, joint] + joint * signal_jump,
                color=colors[joint],
                linewidth=2,
            )

        y_min, y_max = axs[chain].get_ylim()
        axs[chain].set_ylim(y_max, y_min)
        axs[chain].set_ylabel(names_chains[chain])
        axs[chain].set_yticks([])
        axs[chain].grid()

    return


def _position_signal_single_chain(
    frequency: float,
    times: np.ndarray[float],
    amp_joints: np.ndarray[float],
    off_joints: np.ndarray[float],
    bsl_joints: np.ndarray[float],
    ipl_joints: np.ndarray[float],
    sig_function: Callable = None,
):
    ''' Neural output for the DOFs of a single chain '''

    assert len(amp_joints) == len(ipl_joints), \
        'amp_joints and ipl_joints should have the same length'
    assert len(amp_joints) == len(off_joints), \
        'amp_joints and off_joints should have the same length'
    assert len(amp_joints) == len(bsl_joints), \
        'amp_joints and bsl_joints should have the same length'

    n_joints = len(amp_joints)
    times_expanded = np.vstack([times] * n_joints).T

    # Signal function
    if sig_function is None:
        def sig_function(phase): return np.sin(2*np.pi * phase)

    chain_signal_aux = sig_function(frequency*times_expanded - ipl_joints)

    # One signal for each joint
    chain_signal = np.array(
        [
            (
                j_bsl +                             # Baseline
                j_off +                             # Offset
                j_amp * chain_signal_aux[:, j_ind]  # Signal
            )
            for j_ind, (j_off, j_amp, j_bsl) in enumerate(zip(off_joints, amp_joints, bsl_joints))
        ]
    ).T

    return chain_signal


def _position_signal_multi_chain(
    times: np.ndarray[float],
    frq_joints_chains: list[np.ndarray[float]],
    amp_joints_chains: list[np.ndarray[float]],
    off_joints_chains: list[np.ndarray[float]],
    bsl_joints_chains: list[np.ndarray[float]],
    ipl_joints_chains: list[np.ndarray[float]],
    ipl_chains: list[float],
    sig_funcion: Callable = None,
):
    ''' Neural output for the DOFs of multiple chains '''

    n_chains = len(ipl_chains)

    assert len(frq_joints_chains) == n_chains, \
        'frq_joints_chains should have length n_chains'
    assert len(amp_joints_chains) == n_chains, \
        'amp_joints_chains should have length n_chains'
    assert len(off_joints_chains) == n_chains, \
        'off_joints_chains should have length n_chains'
    assert len(bsl_joints_chains) == n_chains, \
        'bsl_joints_chains should have length n_chains'
    assert len(ipl_joints_chains) == n_chains, \
        'ipl_joints_chains should have length n_chains'

    multi_chains_signals_separated = [
        _position_signal_single_chain(
            times=times,
            frequency=frq_joints_chains[chain],
            amp_joints=amp_joints_chains[chain],
            off_joints=off_joints_chains[chain],
            bsl_joints=bsl_joints_chains[chain],
            ipl_joints=ipl_joints_chains[chain] + ipl_chains[chain],
            sig_function=sig_funcion
        )
        for chain in range(n_chains)
    ]
    multi_chains_signals = np.concatenate(
        multi_chains_signals_separated, axis=1)

    return multi_chains_signals, multi_chains_signals_separated


def _get_position_signal(
    times: np.ndarray,
    chains_params: dict[str, dict[str, Any]],
    sig_funcion: Callable = None,
):
    ''' Get neural output '''

    ipl_chains = []
    names_chains = []
    frq_joints_chains = []
    amp_joints_chains = []
    off_joints_chains = []
    bsl_joints_chains = []
    ipl_joints_chains = []

    # CHAINS
    for chain_name, chain_pars in chains_params.items():

        if chain_pars['n_joints'] == 0:
            continue

        frequency_chain = chain_pars.get('frequency')

        # AMPLITUDE
        amp_joints_min = chain_pars.get('amp_min')
        amp_joints_max = chain_pars.get('amp_max')
        amp_joints_arr = chain_pars.get('amp_arr')

        if amp_joints_min is not None and amp_joints_max is not None:
            assert amp_joints_arr is None, \
                'Either amp_joints_arr or amp_joints_min and amp_joints_max must be None'
            amp_joints_arr = np.linspace(
                amp_joints_min, amp_joints_max, chain_pars['n_joints'])

        # OFFSET
        off_joints_min = chain_pars.get('off_min')
        off_joints_max = chain_pars.get('off_max')
        off_joints_arr = chain_pars.get('off_arr')

        if off_joints_min is not None and off_joints_max is not None:
            assert off_joints_arr is None, \
                'Either off_joints_arr or off_joints_min and off_joints_max must be None'
            off_joints_arr = np.linspace(
                off_joints_min, off_joints_max, chain_pars['n_joints'])

        # BASELINE
        bsl_joints_min = chain_pars.get('bsl_min')
        bsl_joints_max = chain_pars.get('bsl_max')
        bsl_joints_arr = chain_pars.get('bsl_arr')

        if bsl_joints_min is not None and bsl_joints_max is not None:
            assert bsl_joints_arr is None, \
                'Either bsl_joints_arr or bsl_joints_min and bsl_joints_max must be None'
            bsl_joints_arr = np.linspace(
                bsl_joints_min, bsl_joints_max, chain_pars['n_joints'])

        # PHASE
        ipl_joints_min = chain_pars.get('ipl_min')
        ipl_joints_max = chain_pars.get('ipl_max')
        ipl_joints_arr = chain_pars.get('ipl_arr')

        if ipl_joints_min is not None and ipl_joints_max is not None:
            assert ipl_joints_arr is None, \
                'Either ipl_joints_arr or ipl_joints_min and ipl_joints_max must be None'
            ipl_joints_arr = np.linspace(
                ipl_joints_min, ipl_joints_max, chain_pars['n_joints'])

        # COPIES
        n_copies = chain_pars.get('n_copies', 1)
        ipl_off = chain_pars.get('ipl_off', [0.0])
        names_copies = chain_pars.get('names', [chain_name.upper()])

        assert len(ipl_off) == n_copies, \
            'ipl_off must have the same length as n_copies'
        assert len(names_copies) == n_copies, \
            'names_copies must have the same length as n_copies'

        # APPEND
        ipl_chains = ipl_chains + ipl_off
        names_chains = names_chains + names_copies
        frq_joints_chains = frq_joints_chains + [frequency_chain] * n_copies
        amp_joints_chains = amp_joints_chains + [amp_joints_arr] * n_copies
        off_joints_chains = off_joints_chains + [off_joints_arr] * n_copies
        bsl_joints_chains = bsl_joints_chains + [bsl_joints_arr] * n_copies
        ipl_joints_chains = ipl_joints_chains + [ipl_joints_arr] * n_copies

    # MOTOR OUTPUT SIGNALS
    motor_output_signals, multi_chains_signals_separated = _position_signal_multi_chain(
        times=times,
        frq_joints_chains=frq_joints_chains,
        amp_joints_chains=amp_joints_chains,
        off_joints_chains=off_joints_chains,
        bsl_joints_chains=bsl_joints_chains,
        ipl_joints_chains=ipl_joints_chains,
        ipl_chains=ipl_chains,
        sig_funcion=sig_funcion,
    )

    return motor_output_signals, multi_chains_signals_separated, names_chains


def define_entraining_signals(
    n_iterations,
    frequency,
    amplitude_degrees=45,
    n_joints=13,
    timestep=0.001,
    plot_signals=False,
):
    ''' Define the entraining signal for the simulation '''

    # Topology
    amp_rad = np.deg2rad(amplitude_degrees) / n_joints

    # Simulation
    times = np.arange(n_iterations) * timestep

    # Signal properties
    motor_output_signal_pars = {
        'axis': {
            'frequency': frequency,
            'n_copies': 1,
            'ipl_off': [0.0],
            'names': ['AXIS'],

            'n_joints': n_joints,
            'ipl_min': 0.0,
            'ipl_max': 0.0,
            'amp_min': amp_rad,
            'amp_max': amp_rad,
            'off_arr': np.zeros(n_joints),
            'bsl_arr': np.zeros(n_joints),
        },
    }

    (
        motor_output_signals,
        multi_chains_signals_separated,
        names_chains,
    ) = _get_position_signal(
        times=times,
        chains_params=motor_output_signal_pars,
    )

    if plot_signals:
        _plot_signals(
            times,
            multi_chains_signals_separated,
            names_chains=names_chains,
        )

    return motor_output_signals


def run_example():

    # Topology
    n_joints_axis = 13

    # Simulation
    duration = 10.0
    timestep = 0.001
    times = np.arange(0.0, duration, timestep)

    # Signal function
    # NOTE: The signal should be a function of phase as t is internally scaled by the frequency
    # NOTE: Setting the frequency to 1.0 makes the signal a function of time
    # (f*t = t)

    def signal_function(phase): return np.sin(2*np.pi * phase)

    # Signal properties
    motor_output_signal_pars = {
        'trunk': {
            'frequency': 1.0,
            'n_copies': 1,
            'ipl_off': [0.0],
            'names': ['AXIS'],

            'n_joints': n_joints_axis,
            'ipl_min': 0.0,
            'ipl_max': 1.0,
            'amp_min': 1.0,
            'amp_max': 1.0,
            'off_arr': np.zeros(n_joints_axis),
            'bsl_arr': np.zeros(n_joints_axis),
        },
    }

    (
        motor_output_signals,
        multi_chains_signals_separated,
        names_chains,
    ) = _get_position_signal(
        times=times,
        chains_params=motor_output_signal_pars,
        sig_funcion=signal_function,
    )

    _plot_signals(
        times,
        multi_chains_signals_separated,
        names_chains=names_chains,
    )

    plt.show()


if __name__ == '__main__':
    run_example()

