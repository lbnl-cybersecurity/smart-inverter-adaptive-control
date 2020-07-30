import numpy as np
from pycigar.controllers.base_controller import BaseController
import copy
from collections import deque

STEP_BUFFER = 4


class CustomAdaptiveInverterController(BaseController):

    def __init__(self, device_id, additional_params):
        """Instantiate an adaptive inverter controller."""
        BaseController.__init__(
            self,
            device_id
        )
        self.lpf_delta_t = additional_params.get('delta_t', 1)
        self.gain = additional_params.get('gain', 1e5)
        self.y_threshold = additional_params.get('y_threshold', 0.03)
        self.k1 = additional_params.get('k1', 1)
        self.k2 = additional_params.get('k2', 1)

        self.init_params = copy.deepcopy(additional_params)
        self.lpf_psi = deque([0]*2, maxlen=2)
        self.lpf_epsilon = deque([0]*2, maxlen=2)
        self.lpf_y1 = deque([0]*2, maxlen=2)
        self.lpf_high_pass_filter = 1
        self.lpf_low_pass_filter = 0.1

        self.x = deque([0]*15, maxlen=15)
        self.y = 0
        self.v_offset = 0

    def get_action(self, env):
        """See parent class."""
        if not hasattr(self, 'node_id'):
            self.node_id = env.k.device.get_node_connected_to(self.device_id)

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))

        if env.k.time > 1:
            vk = abs(env.k.node.nodes[self.node_id]['voltage'][env.k.time - 1])

            #self.x.append(vk)
            if env.k.time >= 16:
                output = abs(env.k.node.nodes[self.node_id]['voltage'][env.k.time-16:env.k.time - 1])
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

            y_value = (self.lpf_delta_t * self.lpf_low_pass_filter *
                    (self.lpf_epsilon[1] + self.lpf_epsilon[0]) - (self.lpf_delta_t * self.lpf_low_pass_filter - 2) * self.lpf_y1[1]) / \
                    (2 + self.lpf_delta_t * self.lpf_low_pass_filter)
            self.lpf_y1.append(y_value)
            self.y = y_value*0.04

            if self.y > self.y_threshold:
                if self.v_offset < 0.05:
                    self.v_offset = -self.k1*self.v_offset + self.k2*self.y
                else:
                    self.v_offset = -self.k1*self.v_offset - self.k2*self.y
            else:
                self.v_offset = -self.k1*self.v_offset

                return None, {'v_offset': self.v_offset}

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
