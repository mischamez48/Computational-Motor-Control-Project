
from metrics import compute_all_metrics, compute_neural_metrics, compute_mechanical_metrics
from util.rw import save_object
from util.mp_util import sweep_1d, sweep_2d
from util.simulation_control import run_experiment
import numpy as np
import json
import os
os.environ['MUJOCO_GL'] = 'glfw'


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pretty(data, indent=4):
    """
    print dictionary d in a pretty format
    """
    print("Computed metrics:")
    print(json.dumps(data, indent=indent, cls=NumpyEncoder))
    return


def run_single(pars):

    animat_data, network = run_experiment(pars)

    # store amimat data vars in network as numpy arrays and delete animat
    network.joints = np.array(animat_data.sensors.joints.array)
    network.links_positions = np.array(
        animat_data.sensors.links.urdf_positions())
    network.links_orientations = np.array(
        animat_data.sensors.links.urdf_orientations())
    network.links_velocities = np.array(
        animat_data.sensors.links.com_lin_velocities())
    network.joints_active_torques = np.array(
        animat_data.sensors.joints.active_torques())
    network.joints_velocities = np.array(
        animat_data.sensors.joints.velocities_all())
    network.joints_positions = np.array(
        animat_data.sensors.joints.positions_all())

    del animat_data

    if pars.compute_metrics == 'neural':
        network.metrics = compute_neural_metrics(network)
    elif pars.compute_metrics == 'mechanical':
        network.metrics = compute_mechanical_metrics(network)
    elif pars.compute_metrics == 'all':
        network.metrics = compute_all_metrics(network)
    elif pars.compute_metrics is None:
        network.metrics = {}
    else:
        raise ValueError(f"Unknown metric type: {pars.compute_metrics}")

    if pars.print_metrics:
        pretty(network.metrics)

    if pars.log_path != "":
        os.makedirs(pars.log_path, exist_ok=True)
        save_object(
            network,
            '{}controller{}'.format(
                pars.log_path,
                pars.simulation_i))

    if pars.return_network:
        return network
    else:
        return None


def run_multiple(pars_list, num_process=6):

    return sweep_1d(run_single, pars_list, num_process=num_process)


def run_multiple2d(pars_list1, pars_list2, num_process=6):

    return sweep_2d(
        run_single,
        pars_list1,
        pars_list2,
        num_process=num_process)

