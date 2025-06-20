"""Descending drive"""

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from simple_pid import PID

from farms_core.io.yaml import yaml2pyobject


class PotentialMap(ABC):
    """Potential map"""

    @abstractmethod
    def heading(self, pos):
        """Heading"""
        raise NotImplementedError

    @staticmethod
    def limit_cycle():
        """Limit cycle"""
        return None

    def heading_cartesian(self, pos, radius=1):
        """Heading cartesian"""
        heading_complex = radius*np.exp(1j*self.heading(pos))
        return heading_complex.real, heading_complex.imag

    def mesh(self, lin_x, lin_y, radius=1):
        """Mesh"""
        dimensions = (len(lin_x), len(lin_y))
        vec_x, vec_y = np.meshgrid(lin_x, lin_y, indexing='ij')
        vec_u, vec_v = np.zeros(dimensions), np.zeros(dimensions)
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                vec_u[i, j], vec_v[i, j] = self.heading_cartesian(
                    pos=np.array([vec_x[i, j], vec_y[i, j]]),
                    radius=radius,
                )
        return vec_x, vec_y, vec_u, vec_v


class StraightLinePotentialMap(PotentialMap):
    """Straight line potential map"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop('gain', 1)
        self.origin = kwargs.pop('origin', np.zeros(2))
        self.theta = kwargs.pop('theta', 0)
        assert not kwargs, kwargs

    def heading(self, pos):
        """Heading"""
        pos_complex = complex(*(pos[:2] - self.origin))
        theta = np.angle(pos_complex)
        r_dot = self.gain*np.linalg.norm(pos_complex)*np.sin(self.theta-theta)
        return np.angle(
            r_dot*np.exp(1j*(self.theta+0.5*np.pi))
            + np.exp(1j*self.theta)
        )

    def limit_cycle(self):
        """Limit cycle"""
        vector_complex = np.exp(1j*self.theta)
        vector = np.array([vector_complex.real, vector_complex.imag])
        return np.array([self.origin + 1e3*vector, self.origin - 1e3*vector])


class CirclePotentialMap(PotentialMap):
    """Circle potential map"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain = kwargs.pop('gain', 1)
        self.origin = kwargs.pop('origin', np.zeros(2))
        self.radius = kwargs.pop('radius', 4)
        self.direction = kwargs.pop('direction', -1)
        assert not kwargs, kwargs

    def heading(self, pos):
        """Heading"""
        pos_complex = complex(*(pos[:2] - self.origin))
        r_dot = self.gain*(self.radius-np.abs(pos_complex))
        theta = np.angle(pos_complex)
        return np.angle(
            r_dot*np.exp(1j*theta)
            + np.exp(1j*(theta+0.5*np.sign(self.direction)*np.pi))
        )

    def limit_cycle(self):
        """Limit cycle"""
        vectors_complex = [
            self.radius*np.exp(1j*theta)
            for theta in np.linspace(0, 2*np.pi, 100)
        ]
        vectors = np.array([[vec.real, vec.imag] for vec in vectors_complex])
        return self.origin + vectors


class DescendingDrive(ABC):
    """Descending drive"""

    def __init__(self, drives):
        super().__init__()
        self.drives = drives
        self.n_drives = np.shape(drives.array)[1]
        self.n_iterations = np.shape(drives.array)[0]
        self.setpoints = np.zeros(self.n_iterations)
        self.control = np.zeros(self.n_iterations)

    @abstractmethod
    def step(self, iteration, time, timestep):
        """Step"""
        raise NotImplementedError

    def get_left_drive(self, iteration):
        """Get forward drive"""
        return self.drives.array[
            min(iteration, self.n_iterations-1),
            self.drives.left_indices[0],
        ]

    def get_right_drive(self, iteration):
        """Get turn drive"""
        return self.drives.array[
            min(iteration, self.n_iterations-1),
            self.drives.right_indices[0],
        ]

    def set_left_drive(self, iteration, values):
        """Set forward drive"""
        for index in self.drives.left_indices:
            self.drives.array[
                min(iteration, self.n_iterations-1),
                index,
            ] = values[index]

    def set_right_drive(self, iteration, values):
        """Set turn drive"""
        for index in self.drives.right_indices:
            self.drives.array[
                min(iteration, self.n_iterations-1),
                index,
            ] = values[index]


class OrientationFollower(DescendingDrive):
    """Descending drive to follow orientation"""

    def __init__(self, strategy, animat_data, timestep, **kwargs):
        self.strategy = strategy
        self.animat_data = animat_data
        super().__init__(drives=animat_data.network.drives)
        self.links_indices = np.array(kwargs.pop('links_indices', [0]))
        self.heading_offset = kwargs.pop('heading_offset', 0)
        self.contact_threshold = kwargs.pop('contact_threshold', 0)
        self.pid = PID(
            Kp=kwargs.pop('pid_p', 0.2),
            Ki=kwargs.pop('pid_i', 0.0),
            Kd=kwargs.pop('pid_d', 0.0),
            sample_time=timestep,
            output_limits=kwargs.pop('output_limits', (-0.9, 0.9)),
        )
        self.contacts_values = np.zeros(self.n_drives)
        self.fwds = np.zeros(self.n_drives)
        self.fwds_raw = np.zeros(self.n_drives)
        self.turn = 0
        assert not kwargs, kwargs

    def update_turn_command(self, pos):
        """Update command"""
        self.pid.setpoint = self.strategy.heading(pos)
        return self.pid.setpoint

    def get_turn_control(self, iteration, timestep, command, heading):
        """Update drive"""
        error = ((command - heading + np.pi) % (2*np.pi)) - np.pi
        return -(self.pid(command-error, dt=timestep))

    def get_foward_control(self, iteration, timestep):
        """Update drive"""
        # xfrc = np.array(
        #     self.animat_data.sensors.xfrc.forces(),
        #     copy=False,
        # )[iteration, indices, :]
        # condition_xfrc = np.count_nonzero(xfrc) < int(0.85*len(indices)*3)
        threshold = 9.81*self.contact_threshold
        contacts = np.array(
            self.animat_data.sensors.contacts.totals()[iteration],
            copy=True,
        )  # [indices]

        # All
        contacts_sum = np.sum(np.abs(contacts))
        self.contacts_values += min(10*timestep, 1)*(
            contacts_sum - self.contacts_values
        )

        self.fwds_raw = np.where(self.contacts_values > threshold, 2, 4)
        return self.fwds_raw

    def update(self, iteration, timestep, pos, heading):
        """Update drive"""
        self.setpoints[iteration] = self.update_turn_command(pos=pos)
        drive_turn = self.get_turn_control(
            iteration=iteration,
            timestep=timestep,
            command=self.setpoints[iteration],
            heading=heading,
        )
        self.turn += min(100*timestep, 1)*(drive_turn - self.turn)
        drive_fwds = self.get_foward_control(
            iteration=iteration,
            timestep=timestep,
        )
        self.fwds += min(100*timestep, 1)*(drive_fwds-self.fwds)
        self.set_left_drive(iteration=iteration, values=self.fwds-self.turn)
        self.set_right_drive(iteration=iteration, values=self.fwds+self.turn)
        self.control[iteration] = self.turn
        return self.setpoints[iteration], self.control[iteration]

    def step(self, iteration, time, timestep):
        """Step"""
        self.update(
            iteration=iteration,
            timestep=timestep,
            pos=np.array(self.animat_data.sensors.links.urdf_position(
                iteration=iteration,
                link_i=self.links_indices[0],
            )),
            heading=self.animat_data.sensors.links.heading(
                iteration=iteration,
                indices=self.links_indices,
            )+self.heading_offset,
        )


class DistributedOrientationFollower(OrientationFollower):
    """Distributed descending drive to follow orientation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intention = np.zeros(self.n_drives)
        self.reaction = np.zeros(self.n_drives)
        self.contact_value = 0

    def get_foward_control(self, iteration, timestep):
        """Update drive"""
        threshold = 9.81*self.contact_threshold
        contacts = np.array(
            self.animat_data.sensors.contacts.totals()[iteration],
            copy=True,
        )

        # Intention
        self.contact_value += min(100*timestep, 1)*(
            np.sum(np.abs(contacts)) - self.contact_value
        )
        self.fwds_raw[:] = 2 if self.contact_value > threshold else 4

        # Reaction
        self.contacts_values += min(100*timestep, 1)*(
            [
                np.sum(np.abs(contacts[indices]))
                for indices in self.drives.contacts_indices
            ] - self.contacts_values
        )

        # Decision
        threshold2 = 1e-3*threshold
        self.fwds_raw[np.logical_and(
            self.fwds_raw > 3,
            self.contacts_values > threshold2,
        )] = 2.9
        self.fwds_raw[np.logical_and(
            self.fwds_raw < 3,
            self.contacts_values < threshold2,
        )] = 3.1

        return self.fwds_raw


def plotting(times, pos, drive, phi):
    """Plotting"""

    # Gain plot
    fig, ax1 = plt.subplots()
    ax1.plot(times, drive.control, label='Control output')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Drive')
    ax1.set_title('PID output')
    ax1.legend()
    plt.grid(True)

    # Orientation plot
    fig2, ax2 = plt.subplots()
    ax2.plot(times, np.degrees(drive.setpoints), label='Orientation setpoint')
    ax2.plot(times, np.degrees(phi), label='Mean orientation')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angle [deg]')
    ax2.set_title('Orientation')
    ax2.legend()
    plt.grid(True)

    # Position plot
    arrow_res = 0.5
    fig3, ax3 = plt.subplots()
    vec_x, vec_y, vec_u, vec_v = drive.strategy.mesh(
        lin_x=np.arange(
            start=np.floor(np.min(pos[:, 0])),
            stop=np.ceil(np.max(pos[:, 0]))+1,
            step=arrow_res,
        ),
        lin_y=np.arange(
            start=np.floor(np.min(pos[:, 1])),
            stop=np.ceil(np.max(pos[:, 1]))+1,
            step=arrow_res,
        ),
        radius=1.5*arrow_res,
    )
    plt.quiver(vec_x, vec_y, vec_u, vec_v, angles='xy')
    x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
    limit_cycle = drive.strategy.limit_cycle()
    if limit_cycle is not None:
        ax3.plot(limit_cycle[:, 0], limit_cycle[:, 1], label='Limit trajectory')
    if pos is not None:
        ax3.plot(pos[0, :], pos[1, :], label='Animat trajectory')
        ax3.plot(pos[0, 0], pos[1, 0], 'x', label='Animat initial position')
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    # if MIXED:
    #     circle1 = plt.Circle(
    #         [3.8, 0],
    #         0.3,
    #         color='r',
    #         fill=False,
    #         label='obstacle',
    #     )
    #     ax3.add_patch(circle1)

    ax3.set_xlabel('X coordinate [m]')
    ax3.set_ylabel('Y coordinate [m]')
    ax3.set_title('Animat trajectory')
    ax3.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')

    return fig, fig2, fig3


def plot_trajectory(strategy, pos, arrow_res=None):
    """Plot trajectory"""
    fig3, ax3 = plt.subplots()
    min_x, max_x, min_y, max_y = (
        np.min(pos[:, 0]),
        np.max(pos[:, 0]),
        np.min(pos[:, 1]),
        np.max(pos[:, 1]),
    )
    if arrow_res is None:
        arrow_res = max(max_x-min_x, max_x-min_x)/30
    vec_x, vec_y, vec_u, vec_v = strategy.mesh(
        lin_x=np.arange(
            start=np.floor(min_x/arrow_res)*arrow_res,
            stop=np.ceil(max_x/arrow_res+1)*arrow_res,
            step=arrow_res,
        ),
        lin_y=np.arange(
            start=np.floor(min_y/arrow_res)*arrow_res,
            stop=np.ceil(max_y/arrow_res+1)*arrow_res,
            step=arrow_res,
        ),
        radius=arrow_res,
    )
    plt.quiver(vec_x, vec_y, vec_u, vec_v, angles='xy')
    x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
    limit_cycle = strategy.limit_cycle()
    if limit_cycle is not None:
        ax3.plot(limit_cycle[:, 0], limit_cycle[:, 1], label='Limit trajectory')
    if pos is not None:
        ax3.plot(pos[:, 0], pos[:, 1], label='Animat trajectory')
        ax3.plot(pos[0, 0], pos[0, 1], 'x', label='Animat initial position')
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    ax3.set_xlabel('X coordinate [m]')
    ax3.set_ylabel('Y coordinate [m]')
    ax3.set_title('Animat trajectory')
    ax3.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')
    return fig3


def assess_traj(traj, t_traj):
    """Assess trajectory"""
    tree = KDTree(np.transpose(t_traj))
    distances, indices = tree.query(np.transpose(traj), k=1)
    return distances, indices, np.sum(distances)


def drive_from_config(filename, animat_data, simulation_options):
    """Drive from config"""
    drive_config = yaml2pyobject(filename)
    potential_config = drive_config.pop('potential_map')
    potential_type = potential_config.pop('type')
    control_function = (
        DistributedOrientationFollower
        if potential_type in ('disline', 'discircle')
        else OrientationFollower
        if potential_type in ('line', 'circle')
        else None
    )
    assert control_function is not None, f'Unknown {potential_type=}'
    return control_function(
        strategy={
            'line': StraightLinePotentialMap,
            'circle': CirclePotentialMap,
            'disline': StraightLinePotentialMap,
            'discircle': CirclePotentialMap,
        }[potential_type](**potential_config),
        animat_data=animat_data,
        timestep=simulation_options.timestep,
        **drive_config,
    )
