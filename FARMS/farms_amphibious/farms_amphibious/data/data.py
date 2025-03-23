"""Amphibious data"""

from typing import Dict

import numpy as np

from farms_core import pylog
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.array.array import to_array
from farms_core.array.array_cy import IntegerArray1D, IntegerArray2D
from farms_core.array.types import NDARRAY_V1
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions, ControlOptions
from farms_core.simulation.options import SimulationOptions
from farms_core.sensors.data import SensorsData

from ..model.options import (
    AmphibiousControlOptions,
    GenericControlOptions,
    KinematicsControlOptions,
)

from .data_cy import (
    AmphibiousDataCy,
    GenericDataCy,
    JointsControlArrayCy
)

from .network import (
    OscillatorNetworkState,
    GenericNetworkState,
    NetworkParameters,
    GenericNetworkParameters,
    DriveArray,
    Oscillators,
    GenericOscillators,
    OscillatorConnectivity,
    JointsConnectivity,
    ContactsConnectivity,
    XfrcConnectivity,
)


def get_amphibious_data(animat_options, simulation_options):
    """Get amphibious_data"""
    return (
        AmphibiousKinematicsData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
        if isinstance(animat_options.control, KinematicsControlOptions)
        else AmphibiousData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
        if isinstance(animat_options.control, AmphibiousControlOptions)
        else GenericData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
        if isinstance(animat_options.control, GenericControlOptions)
        else AnimatData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
    )

class JointsControlArray(JointsControlArrayCy):
    """Oscillator array"""

    @classmethod
    def from_options(cls, control: ControlOptions):
        """Default"""
        return cls(
            array=np.array([
                [
                    offset['gain'],
                    offset['bias'],
                    offset['low'],
                    offset['high'],
                    offset['saturation'],
                    rate,
                ]
                for offset, rate in zip(
                    control.motors_offsets(),
                    control.motors_offset_rates(),
                )
            ], dtype=np.double),
            drive2joint_map=IntegerArray2D(
                np.array(control.network.drive2joint, dtype=np.uintc)
            ),
        )


class AmphibiousData(AmphibiousDataCy, AnimatData):
    """Amphibious data"""

    def __init__(
            self,
            state: OscillatorNetworkState,
            network: NetworkParameters,
            joints: JointsControlArray,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = state
        self.network = network
        self.joints = joints

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From animat and simulation options"""

        # Sensors
        sensors = SensorsData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )

        # Oscillators
        oscillators = Oscillators.from_options(
            network=animat_options.control.network,
        ) if animat_options.control.network is not None else None

        # Maps
        oscillators_map, joints_map, contacts_map, xfrc_map = (
            [
                {
                    name: element_i
                    for element_i, name in enumerate(element.names)
                }
                for element in (
                        oscillators,
                        sensors.joints,
                        sensors.contacts,
                        sensors.xfrc,
                )
            ]
            if animat_options.control.network is not None
            else (None, None, None, None)
        )

        # State
        state = (
            OscillatorNetworkState.from_initial_state(
                initial_state=animat_options.state_init(),
                n_iterations=simulation_options.n_iterations,
                n_oscillators=animat_options.control.network.n_oscillators(),
            )
            if animat_options.control.network is not None
            else None
        )

        # Network
        network = (
            NetworkParameters(
                drives=DriveArray.from_animat_options(
                    animat_options=animat_options,
                    n_iterations=simulation_options.n_iterations,
                ),
                oscillators=oscillators,
                osc2osc_map=OscillatorConnectivity.from_connectivity(
                    connectivity=animat_options.control.network.osc2osc,
                    map1=oscillators_map,
                    map2=oscillators_map,
                ),
                joints2osc_map=JointsConnectivity.from_connectivity(
                    connectivity=animat_options.control.network.joint2osc,
                    map1=oscillators_map,
                    map2=joints_map,
                ),
                contacts2osc_map=(
                    ContactsConnectivity.from_connectivity(
                        connectivity=animat_options.control.network.contact2osc,
                        map1=oscillators_map,
                        map2=contacts_map,
                    )
                ),
                xfrc2osc_map=XfrcConnectivity.from_connectivity(
                    connectivity=animat_options.control.network.xfrc2osc,
                    map1=oscillators_map,
                    map2=xfrc_map,
                ),
            )
            if animat_options.control.network is not None
            else None
        )

        return cls(
            timestep=simulation_options.timestep,
            sensors=sensors,
            state=state,
            network=network,
            joints=JointsControlArray.from_options(animat_options.control),
        )

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        data['n_oscillators'] = len(data['network']['oscillators']['names'])
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        n_oscillators = dictionary.pop('n_oscillators')
        return cls(
            timestep=dictionary['timestep'],
            state=OscillatorNetworkState(dictionary['state'], n_oscillators),
            network=NetworkParameters.from_dict(dictionary['network']),
            joints=JointsControlArrayCy(
                array=dictionary['joints'],
                drive2joint_map=IntegerArray2D(dictionary['drive2joint_map']),
            ),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        data_dict = super().to_dict(iteration=iteration)
        data_dict.update({
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
            'joints': to_array(self.joints.array),
            'drive2joint_map': to_array(self.joints.drive2joint_map.array),
        })
        return data_dict

    def plot(self, times: NDARRAY_V1) -> Dict:
        """Plot"""
        plots = {}
        plots.update(self.state.plot(times))
        plots.update(self.plot_sensors(times))
        plots['drives'] = self.network.drives.plot(times)
        return plots


class AmphibiousKinematicsData(AnimatData):
    """Amphibious data"""

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From animat and simulation options"""
        return cls(
            timestep=simulation_options.timestep,
            sensors=SensorsData.from_options(
                animat_options=animat_options,
                simulation_options=simulation_options,
            ),
        )

# --------------------- [ Generic ] ---------------------
class GenericData(GenericDataCy, AnimatData):
    """Generic data"""

    def __init__(
            self,
            state: GenericNetworkState,
            network: GenericNetworkParameters,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = state
        self.network = network

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From animat and simulation options"""

        # Sensors
        sensors = SensorsData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )

        # Oscillators
        oscillators = GenericOscillators.from_options(
            network=animat_options.control.network,
        ) if animat_options.control.network is not None else None


        # State
        state = (
            GenericNetworkState.from_initial_state(
                initial_state=animat_options.state_init(),
                n_iterations=simulation_options.n_iterations,
                n_oscillators=animat_options.control.network.n_oscillators(),
            )
            if animat_options.control.network is not None
            else None
        )

        # Network
        network = (
            GenericNetworkParameters(oscillators=oscillators)
            if animat_options.control.network is not None
            else None
        )

        return cls(
            timestep=simulation_options.timestep,
            sensors=sensors,
            state=state,
            network=network,
        )

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        data['n_oscillators'] = len(data['network']['oscillators']['names'])
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        n_oscillators = dictionary.pop('n_oscillators')
        return cls(
            timestep=dictionary['timestep'],
            state=GenericNetworkState(dictionary['state'], n_oscillators),
            network=GenericNetworkParameters.from_dict(dictionary['network']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        data_dict = super().to_dict(iteration=iteration)
        data_dict.update({
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
        })
        return data_dict

# \-------------------- [ Generic ] ---------------------