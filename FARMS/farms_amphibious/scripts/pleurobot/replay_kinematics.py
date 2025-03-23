#!/usr/bin/env python3
"""Run pleurobot simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_models.utils import get_model_path, get_sdf_path
from farms_core.utils.profile import profile
from farms_core.simulation.options import SimulationOptions
from farms_amphibious.utils.prompt import prompt
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.experiment.simulation import simulation
from farms_amphibious.experiment.options import (
    get_pleurobot_kwargs_options,
    get_pleurobot_options,
    amphibious_options,
)


def main():
    """Main"""

    kwargs = get_pleurobot_kwargs_options(
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi],
        kinematics_file=os.path.join(
            get_model_path('pleurobot', '1'),
            'kinematics',
            'kinematics.csv',
        ),
        kinematics_sampling=5e-2,  # [s]
    )
    animat_options = get_pleurobot_options(**kwargs, passive=False)
    sdf = get_sdf_path('pleurobot', '1')

    # Torque limits
    for joint in animat_options.control.joints:
        joint.max_torque = 20

    (
        simulation_options,
        arena,
    ) = amphibious_options(animat_options)

    # Save options
    animat_options_filename = 'pleurobot_animat_options.yaml'
    animat_options.save(animat_options_filename)
    simulation_options_filename = 'pleurobot_simulation_options.yaml'
    simulation_options.save(simulation_options_filename)

    # Load options
    animat_options = AmphibiousOptions.load(animat_options_filename)
    simulation_options = SimulationOptions.load(simulation_options_filename)

    # Simulation
    sim = profile(
        function=simulation,
        animat_sdf=sdf,
        animat_options=animat_options,
        simulation_options=simulation_options,
        arena=arena,
        use_controller=True,
    )

    # Post-processing
    pylog.info('Simulation post-processing')
    log_path = 'pleurobot_results'
    video_name = os.path.join(log_path, 'simulation.mp4')
    if log_path and not os.path.isdir(log_path):
        os.mkdir(log_path)
    sim.postprocess(
        iteration=sim.iteration,
        log_path=log_path if prompt('Save data', False) else '',
        plot=prompt('Show plots', False),
        video=video_name if sim.options.record else '',
    )

    # Plot network
    if prompt('Show connectivity maps', False):
        plot_networks_maps(animat_options.morphology, sim.animat().data)

    # Plot
    plt.show()


if __name__ == '__main__':
    TIC = time.time()
    main()
    pylog.info('Total simulation time: {} [s]'.format(time.time() - TIC))
