from collections import deque
from pycigar.devices import PVDevice
import numpy as np

import pycigar.utils.signal_processing as signal_processing
from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

DEFAULT_CONTROL_SETTING = [0.98, 1.01, 1.02, 1.05, 1.07]
STEP_BUFFER = 4


class CustomPVDevice(PVDevice):
    def __init__(self, device_id, additional_params):
        """Instantiate an PV device."""
        PVDevice.__init__(
            self,
            device_id,
            additional_params
        )

        self.custom_control_setting = {}

    def update(self, k):
        """See parent class."""
        VBP = self.control_setting
        # record voltage magnitude measurement
        if not hasattr(self, 'node_id'):
            self.node_id = k.device.get_node_connected_to(self.device_id)
        k.node.nodes[self.node_id]['PQ_injection']['P'] += 100
        k.node.nodes[self.node_id]['PQ_injection']['Q'] += 100

        # log necessary info
        self.log()

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
        self.log()

    def set_control_setting(self, control_setting, custom_control_setting=None):
        """See parent class."""
        if control_setting is not None:
            self.control_setting = control_setting
        if custom_control_setting:
            self.custom_control_setting = custom_control_setting

    def log(self):
        # log history
        Logger = logger()
        Logger.log(self.device_id, 'dummy_log_value', 0)
