from collections import deque
import numpy as np

from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

STEP_BUFFER = 4

class DirectPowerInjectionDevice(BaseDevice):
    def __init__(self, device_id, additional_params):
        """Instantiate an PV device."""
        BaseDevice.__init__(
            self,
            device_id,
            additional_params
        )

        self.custom_control_setting = {}
        self.q_inj = 0

        self.T= additional_params.get('delta_t', 1)
        self.low_pass_filter_measure_mean = additional_params.get('low_pass_filter_measure_mean', 1.0)
        self.low_pass_filter_measure_std = additional_params.get('low_pass_filter_measure_std', 0)
        self.lpf_m = self.low_pass_filter_measure_std * np.random.randn() + self.low_pass_filter_measure_mean

        self.low_pass_filter_v= deque([0]*2, maxlen=2)


    def update(self, k):
        # record voltage magnitude measurement
        if not hasattr(self, 'node_id'):
            self.node_id = k.device.get_node_connected_to(self.device_id)

        if k.time > 1:

            #compute low pass filter of measured voltage: v_meas
            #get voltages at present and past timestep
            vk = abs(k.node.nodes[self.node_id]['voltage'][k.time - 1])
            vkm1 = abs(k.node.nodes[self.node_id]['voltage'][k.time - 2])

            #compute low pass filtered (measured) voltage
            v_m = (T * self.lpf_m * (vk + vkm1) -
                                (T * self.lpf_m - 2) * (self.low_pass_filter_v[1])) / \
                                (2 + T * lpf_m)

            #store measured voltage
            self.low_pass_filter_v.append(v_m)

            #determine power injection from MRAC adaptive controller
            if 'pow_inject' in self.custom_control_setting:
                self.q_inj = self.self.custom_control_setting['pow_inject']
                k.node.nodes[self.node_id]['PQ_injection']['Q'] += self.q_inj

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
        Logger.log(self.device_id, 'q_inj', self.q_inj)
