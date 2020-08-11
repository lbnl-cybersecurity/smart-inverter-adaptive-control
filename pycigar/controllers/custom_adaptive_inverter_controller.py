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
        self.y_threshold = additional_params.get('y_threshold', 0.02)
        self.gamma = additional_params.get('gamma', 1e-3)
        self.k = additional_params.get('k', 50)
        self.delta_t = additional_params.get('delta_t', 1)
        
        self.init_params = copy.deepcopy(additional_params)
        self.lpf_vT = deque([0]*2, maxlen=2)
        self.lpf_vAvg = deque([0]*2, maxlen=2)
        self.lpf_m = additional_params.get('lpf_m', 1)
        self.lpf_avg = additional_params.get('lpf_avg', 10)

        self.x = deque([0]*15, maxlen=15)
        self.y = 0
        self.v_offset = 0
        self.dir = 1
        
    def get_action(self, env):
        """See parent class."""
        T = self.delta_t
        if not hasattr(self, 'node_id'):
            self.node_id = env.k.device.get_node_connected_to(self.device_id)

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))


        vk = abs(env.k.node.nodes[self.node_id]['voltage'][env.k.time - 1])
        vkm1 = abs(env.k.node.nodes[self.node_id]['voltage'][env.k.time - 2])


        vt = (T * self.lpf_m * (vk + vkm1) -(T * self.lpf_m - 2) * (self.lpf_vT[1])) / \
                (2 + T * self.lpf_m)

        self.lpf_vT.append(vt)

        vAvg = (T * self.lpf_avg * (vk + vkm1) -(T * self.lpf_avg - 2) * (self.lpf_vAvg[1])) / \
                (2 + T * self.lpf_avg)

        self.lpf_vAvg.append(vAvg)
        
        dir = 1
        
        if env.k.time > 100:
            #if (vt-vAvg)**2 > self.y_threshold:
            if (vAvg+self.v_offset) > 1.08 and self.dir > 0:
                self.dir = -2
            elif vAvg+self.v_offset < 0.97:
                self.dir = 1

                
            self.v_offset = self.v_offset-self.gamma*self.v_offset + self.dir*self.k*(vt-vAvg)**2

        else:
            self.v_offset=0
        return None, {'v_offset': self.v_offset}

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
