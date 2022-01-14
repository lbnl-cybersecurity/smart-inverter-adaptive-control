from collections import deque

import numpy as np

from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

DEFAULT_CONTROL_SETTING = [0.98, 1.01, 1.02, 1.05, 1.07]
STEP_BUFFER = 4


class PVDevice(BaseDevice):
    def __init__(self, device_id, additional_params):
        """Instantiate an PV device."""
        BaseDevice.__init__(
            self,
            device_id,
            additional_params
        )
        self.solar_generation = None
        self.Sbar = None

        self.control_setting = additional_params.get('default_control_setting', DEFAULT_CONTROL_SETTING)
        self.low_pass_filter_measure_mean = additional_params.get('low_pass_filter_measure_mean', 1.2)
        self.low_pass_filter_output_mean = additional_params.get('low_pass_filter_output_mean', 0.1)
        self.low_pass_filter_measure_std = additional_params.get('low_pass_filter_measure_std', 0)
        self.low_pass_filter_output_std = additional_params.get('low_pass_filter_output_std', 0)

        self.low_pass_filter_measure = self.low_pass_filter_measure_std * np.random.randn() + self.low_pass_filter_measure_mean
        self.low_pass_filter_output = self.low_pass_filter_output_std * np.random.randn() + self.low_pass_filter_output_mean

        self.low_pass_filter = additional_params.get('low_pass_filter', 0.1)
        self.high_pass_filter = additional_params.get('high_pass_filter', 1)
        self.gain = additional_params.get('adaptive_gain', 1e5)
        self.delta_t = additional_params.get('delta_t', 1)
        self.solar_min_value = additional_params.get('solar_min_value', 0)

        Logger = logger()
        if 'init_control_settings' in Logger.custom_metrics:
            logger().custom_metrics['init_control_settings'][device_id] = self.control_setting
        else:
            logger().custom_metrics['init_control_settings'] = {device_id: self.control_setting}

        self.p_set = deque([0]*2, maxlen=2)
        self.q_set = deque([0]*2, maxlen=2)
        self.p_out = deque([0]*2, maxlen=2)
        self.q_out = deque([0]*2, maxlen=2)
        self.low_pass_filter_v = deque([0]*2, maxlen=2)

        self.solar_irr = 0
        self.psi = 0
        self.epsilon = 0
        self.y = 0
        self.u = 0

        self.voltage_state = -1
        self.q_bar = 0

        self.custom_control_setting = {}

        # init for signal processing on Voltage
        self.lpf_psi = deque([0]*2, maxlen=2)
        self.lpf_epsilon = deque([0]*2, maxlen=2)
        self.lpf_y1 = deque([0]*2, maxlen=2)
        self.lpf_high_pass_filter = 1
        self.lpf_low_pass_filter = 0.1
        self.lpf_x = deque([0]*15, maxlen=15)
        self.lpf_delta_t = self.delta_t
        self.lpf_y = 0
        self.x = deque([0]*15, maxlen=15)

    def update(self, k):
        """See parent class."""
        VBP = self.control_setting

        # record voltage magnitude measurement
        if not hasattr(self, 'node_id'):
            self.node_id = k.device.get_node_connected_to(self.device_id)

        self.solar_irr = self.solar_generation[k.time]

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))

        if k.time > 1:
            vk = abs(k.node.nodes[self.node_id]['voltage'][k.time - 1])
            vkm1 = abs(k.node.nodes[self.node_id]['voltage'][k.time - 2])

            #self.x.append(vk)
            if k.time >= 16:
                output = abs(k.node.nodes[self.node_id]['voltage'][k.time-16:k.time - 1])
            else:
                self.x.append(vk)
                output = np.array(self.x)

            if np.max(output[STEP_BUFFER:-STEP_BUFFER]) - np.min(output[STEP_BUFFER:-STEP_BUFFER]) > 0.004:
                norm_data = -1 + 2 * (output - np.min(output)) / (np.max(output) - np.min(output))
                step_corr = np.convolve(norm_data, self.step, mode='valid')

                if max(abs(step_corr)) > 10:
                    output = np.ones(15)
            filter_data = output[STEP_BUFFER:-STEP_BUFFER]

            lpf_psik = (filter_data[-1] - filter_data[-2] - (self.lpf_high_pass_filter * self.lpf_delta_t / 2 - 1) * self.lpf_psi[1]) / \
                            (1 + self.lpf_high_pass_filter * self.lpf_delta_t / 2)
            self.lpf_psi.append(lpf_psik)

            lpf_epsilonk = self.gain * (lpf_psik ** 2)
            self.lpf_epsilon.append(lpf_epsilonk)

        T = self.delta_t
        lpf_m = self.low_pass_filter_measure
        lpf_o = self.low_pass_filter_output

        pk = 0
        qk = 0

        self.voltage_state = -1
        #integer representing where the voltage is in the VVC/VWC
        self.q_bar = 0
        #variable to log the available reactive power

        if k.time > 1:
            low_pass_filter_v = (T * lpf_m * (vk + vkm1) -
                                (T * lpf_m - 2) * (self.low_pass_filter_v[1])) / \
                                (2 + T * lpf_m)

            self.low_pass_filter_v.append(low_pass_filter_v)

            if 'v_offset' in self.custom_control_setting:
                low_pass_filter_v += self.custom_control_setting['v_offset']
                #self.y = self.custom_control_setting['v_offset']
            # compute p_set and q_set
            if self.solar_irr >= self.solar_min_value:
                if low_pass_filter_v <= VBP[4]:
                    # no curtailment
                    pk = -self.solar_irr
                    q_avail = (self.Sbar ** 2 - pk ** 2) ** (1 / 2)

                    self.q_bar = q_avail

                    # determine VAR support
                    if low_pass_filter_v <= VBP[0]:
                        # inject all available var
                        qk = -q_avail
                        self.voltage_state = 0
                    elif VBP[0] < low_pass_filter_v <= VBP[1]:
                        # partial VAR injection
                        c = q_avail / (VBP[1] - VBP[0])
                        qk = c * (low_pass_filter_v - VBP[1])
                        self.voltage_state = 1
                    elif VBP[1] < low_pass_filter_v <= VBP[2]:
                        # No var support
                        qk = 0
                        voltage_state = 2
                    elif VBP[2] < low_pass_filter_v < VBP[3]:
                        # partial Var consumption
                        c = q_avail / (VBP[3] - VBP[2])
                        qk = c * (low_pass_filter_v - VBP[2])
                        self.voltage_state = 3
                    elif VBP[3] < low_pass_filter_v < VBP[4]:
                        # partial real power curtailment
                        d = -self.solar_irr / (VBP[4] - VBP[3])
                        pk = d * (VBP[4] - low_pass_filter_v)
                        qk = (self.Sbar ** 2 - pk ** 2) ** (1 / 2)
                        self.voltage_state = 4
                elif low_pass_filter_v >= VBP[4]:
                    # full real power curtailment for VAR support
                    pk = 0
                    qk = self.Sbar
                    self.voltage_state = 5

            self.p_set.append(pk)
            self.q_set.append(qk)

            # compute p_out and q_out
            self.p_out.append((T * lpf_o * (self.p_set[1] + self.p_set[0]) - (T * lpf_o - 2) * (self.p_out[1])) / \
                            (2 + T * lpf_o))
            self.q_out.append((T * lpf_o * (self.q_set[1] + self.q_set[0]) - (T * lpf_o - 2) * (self.q_out[1])) / \
                            (2 + T * lpf_o))


        # import old V to x
        # inject to node
        k.node.nodes[self.node_id]['PQ_injection']['P'] += self.p_out[1]
        k.node.nodes[self.node_id]['PQ_injection']['Q'] += self.q_out[1]

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
        Logger.log(self.device_id, 'y', self.y)
        Logger.log(self.device_id, 'voltage_state', self.voltage_state)
        Logger.log(self.device_id, 'q_bar', self.q_bar)

        Logger.log(self.device_id, 'u', self.u)
        Logger.log(self.device_id, 'p_set', self.p_set[1])
        Logger.log(self.device_id, 'q_set', self.q_set[1])
        Logger.log(self.device_id, 'p_out', self.p_out[1])
        Logger.log(self.device_id, 'q_out', self.q_out[1])
        Logger.log(self.device_id, 'control_setting', self.control_setting)
        if self.Sbar:
            Logger.log(self.device_id, 'sbar_solarirr', 1.5e-3*(abs(self.Sbar ** 2 - max(10, self.solar_irr) ** 2)) ** (1 / 2))
            Logger.log(self.device_id, 'sbar_pset', self.p_set[1] / self.Sbar)

        Logger.log(self.device_id, 'solar_irr', self.solar_irr)
        if hasattr(self, 'node_id'):
            Logger.log_single(self.device_id, 'node', self.node_id)
