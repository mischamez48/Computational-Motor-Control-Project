"""Prompt"""

import os
from distutils.util import strtobool
import matplotlib.pyplot as plt
from farms_core import pylog
from farms_core.simulation.options import Simulator
from farms_amphibious.utils.network import plot_networks_maps
from ..data.data import AmphibiousData


def prompt(query, default):
    """Prompt"""
    val = input(f'{query} [{"Y/n" if default else "y/N"}]: ')
    try:
        ret = strtobool(val) if val != '' else default
    except ValueError:
        pylog.error('Did not recognise \'%s\', please reply with a y/n', val)
        return prompt(query, default)
    return ret


def prompt_postprocessing(sim, animat_options, query=True, **kwargs):
    """Prompt postprocessing"""
    # Arguments
    log_path = kwargs.pop('log_path', '')
    verify = kwargs.pop('verify', False)
    extension = kwargs.pop('extension', 'pdf')
    simulator = kwargs.pop('simulator', Simulator.MUJOCO)
    assert not kwargs, kwargs

    # Post-processing
    pylog.info('Simulation post-processing')
    save_data = (
        (query and prompt('Save data', False))
        or log_path and not query
    )
    if log_path:
        os.makedirs(log_path, exist_ok=True)
    show_plots = prompt('Show plots', False) if query else False
    iteration = (
        sim.iteration
        if simulator == Simulator.PYBULLET
        else sim.task.iteration  # Simulator.MUJOCO
    )
    sim.postprocess(
        iteration=iteration,
        log_path=log_path if save_data else '',
        plot=show_plots,
        video=(
            os.path.join(log_path, 'simulation.mp4')
            if sim.options.record
            else ''
        ),
    )
    if save_data and verify:
        pylog.debug('Data saved, now loading back to check validity')
        AmphibiousData.from_file(os.path.join(log_path, 'simulation.hdf5'))
        pylog.debug('Data successfully saved and logged back')

    # Save MuJoCo MJCF
    if simulator == Simulator.MUJOCO:
        sim.save_mjcf_xml(os.path.join(log_path, 'sim_mjcf.xml'))

    # Plot network
    show_connectivity = (
        prompt('Show connectivity maps', False)
        if query
        else False
    )
    if show_connectivity:
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Save plots
    if (
            (show_plots or show_connectivity)
            and query
            and prompt('Save plots', False)
    ):
        for fig in [plt.figure(num) for num in plt.get_fignums()]:
            path = os.path.join(log_path, fig.canvas.get_window_title())
            filename = f'{path}.{extension}'
            filename = filename.replace(' ', '_')
            pylog.debug('Saving to %s', filename)
            fig.savefig(filename, format=extension)

    # Show plots
    if show_plots or (
            show_connectivity
            and query
            and prompt('Show connectivity plots', False)
    ):
        plt.show()
