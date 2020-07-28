from pycigar.core.kernel.simulation import OpenDSSSimulation
from pycigar.core.kernel.scenario import OpenDSSScenario
from pycigar.core.kernel.node import OpenDSSNode
from pycigar.core.kernel.device import OpenDSSDevice

from pycigar.utils.exeptions import FatalPyCIGARError
import numpy as np
import random
import pandas as pd
from pycigar.utils.logging import logger
from pycigar.devices.vectorized_pv_inverter_device import VectorizedPVDevice


class Kernel(object):
    """Kernel for abstract function calling across grid simulator APIs.

    The kernel contains four different subclasses for distinguishing between
    the various components of a grid simulator.
    * simulation: controls starting, loading, saving, advancing, and resetting
      the simulator (see pycigar/core/kernel/simulation/base.py)
    * scenario: generates components for an experiment. (see
      pycigar/core/kernel/scenario/base.py)
    * device: stores, regularly updates device information, apply control
      on devices. (see pycigar/core/kernel/device/base.py).
    * node: stores and regularly updates node information
      (see pycigar/core/kernel/node/base.py).

    The above kernel subclasses are designed specifically to support
    simulator-agnostic state information calling. For example, if you would
    like to collect the control setting of a specific device, then simply type:
    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> device_id = "..."  # some device ID
    >>> control_setting = k.device.get_control_setting(device_id)
    In addition, these subclasses support sending commands to the simulator via
    its API. For example, in order to assign a specific vehicle a target
    acceleration, type:
    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> device_id = "..."  # some device ID
    >>> control_setting = "..."  # some device ID
    >>> k.device.apply_cotrol(device_id, control_setting)

    These subclasses can be modified and recycled to support various different
    grid simulators, e.g. OpenDSS, Gridlab-D & OMF...
    """

    def __init__(self, simulator, sim_params):
        """Instantiate a PyCIGAR kernel object.

        Parameters
        ----------
        simulator : string
            The name of simulator we would like to use. For now, there is only
            "opendss".
        sim_params : dict
            The simulation parameters of the experiment.

        Raises
        ------
        FatalPyCIGARError
            The simulator is unkown.
        """
        self.kernel_api = None
        self.sim_params = sim_params
        if self.sim_params != None:
            self.multi_config = self.sim_params['scenario_config']['multi_config']
        self.time = 0
        # initialize logger
        logger()

        self.data_length = pd.read_csv(self.sim_params['scenario_config']['network_data_directory']).shape[0] - 1

        if simulator == "opendss":
            self.simulation = OpenDSSSimulation(self)
            self.scenario = OpenDSSScenario(self)
            self.node = OpenDSSNode(self)
            self.device = OpenDSSDevice(self)
        else:
            raise FatalPyCIGARError("Simulator type '{}' is not valid.".format(simulator))

    def pass_api(self, kernel_api):
        """Pass the API to kernel subclasses."""
        self.kernel_api = kernel_api
        self.simulation.pass_api(kernel_api)
        self.scenario.pass_api(kernel_api)
        self.node.pass_api(kernel_api)
        self.device.pass_api(kernel_api)

    def update(self, reset):
        """Call update for each simulator step.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step.
        """
        if reset is True:
            # reset logger here
            logger().reset()
            logger().set_active(False)
            # track substation, output specs
            # start and end times are given
            if isinstance(self.sim_params['scenario_config']['start_end_time'], list):
                start_time, end_time = self.sim_params['scenario_config']['start_end_time']
                self.t = end_time - start_time

            elif self.multi_config is False:
                self.t = self.sim_params['scenario_config']['start_end_time']
                if 'start_time' not in self.sim_params['scenario_config']:
                    # generate random times (once)
                    start_time = random.randint(0, self.data_length - self.t)
                    end_time = start_time + self.t
                else:
                    # restore previous times
                    start_time = self.sim_params['scenario_config']['start_time']
                    end_time = self.sim_params['scenario_config']['end_time']

            else:
                # generate random times (everytimes)
                self.t = self.sim_params['scenario_config']['start_end_time']
                start_time = random.randint(0, self.data_length - self.t)
                end_time = start_time + self.t

            self.sim_params['scenario_config']['start_time'] = start_time
            self.sim_params['scenario_config']['end_time'] = end_time

            self.time = 0
            self.device.start_device()
            self.scenario.start_scenario()
            self.device.update(reset)

            start_time = self.sim_params['scenario_config']['start_time']
            end_time = self.sim_params['scenario_config']['end_time']
            self.scenario.change_load_profile(start_time, end_time)

            self.node.update(reset)
            self.simulation.update(reset)
            self.scenario.update(reset)

            if self.sim_params['vectorized_mode']:
                self.device.vectorized_pv_inverter_device.reset()

            self.warm_up_k_step(50)
            logger().set_active()

        else:
            self.device.update(reset)  # calculate new PQ with new VBP, then push PV to node
            self.node.update(reset)  # with the load, update the load-pq to simulator
            self.simulation.update(reset)  # run a simulation step
            self.scenario.update(reset)  # update voltage on node
            self.log()

        self.time += 1

    def close(self):
        """Close the simulation and simulator."""
        self.simulation.close()

    def warm_up_1_step(self):
        self.time += 1
        self.device.update(reset=False)
        self.node.update(reset=False)
        self.simulation.update(reset=False)
        self.scenario.update(reset=False)

    def warm_up_k_step(self, k):

        for _ in range(k):
            self.time += 1
            self.device.update(reset=False)
            self.node.update(reset=False)
            self.simulation.update(reset=False)
            self.scenario.update(reset=False)

    def warm_up_v(self):
        """Run the simulation until the voltage is stablized."""

        voltages = self.node.get_all_nodes_voltage()
        self.time += 1
        self.device.update(reset=False)
        self.node.update(reset=False)
        self.simulation.update(reset=False)
        self.scenario.update(reset=False)
        while any(abs(deltaV) > 7e-4 for deltaV in np.array(self.node.get_all_nodes_voltage()) - np.array(voltages)):
            voltages = self.node.get_all_nodes_voltage()
            self.time += 1
            self.device.update(reset=False)
            self.node.update(reset=False)
            self.simulation.update(reset=False)
            self.scenario.update(reset=False)

    def warm_up_y(self):
        """Run the simulation until the voltage is stablized."""
        device_ids = self.device.get_adaptive_device_ids() + self.device.get_fixed_device_ids()
        y = []
        for device_id in device_ids:
            y.append(self.device.get_device_y(device_id))

        self.time += 1
        self.device.update(reset=False)
        self.node.update(reset=False)
        self.simulation.update(reset=False)
        self.scenario.update(reset=False)

        newy = []
        for device_id in device_ids:
            newy.append(self.device.get_device_y(device_id))
        deltay = np.array(np.array(newy) - np.array(y))

        while any(abs(deltay) > 1e-6) or all(deltay == 0):
            y = newy
            self.time += 1
            self.device.update(reset=False)
            self.node.update(reset=False)
            self.simulation.update(reset=False)
            self.scenario.update(reset=False)
            newy = []
            for device_id in device_ids:
                newy.append(self.device.get_device_y(device_id))

            deltay = np.array(np.array(newy) - np.array(y))

    def log(self):
        Logger = logger()
        Logger.log('network', 'substation_power', self.kernel_api.get_total_power())
        Logger.log('network', 'loss', self.kernel_api.get_losses())
        Logger.log('network', 'substation_top_voltage', self.kernel_api.get_substation_top_voltage())
        Logger.log('network', 'substation_bottom_voltage', self.kernel_api.get_substation_bottom_voltage())
