import numpy as np
import numpy.linalg as LA
from pycigar.controllers.base_controller import BaseController
import copy
from collections import deque

class MRAC_adaptive_controller(BaseController):

    def __init__(self, device_id, additional_params):
        """Instantiate an adaptive inverter controller."""
        BaseController.__init__(
            self,
            device_id
        )
        #need a copy of these for reset method
        self.init_params = copy.deepcopy(additional_params)

        ####Controller parameters###

        #reference voltage LPF critical frequency
        self.lpf_ref = additional_params.get('measurement filter time constant avg v', 10)
        #gain on the adaptation law
        self.gamma = additional_params.get('gamma', 1e-3)
        #simulation timestep
        self.delta_t = additional_params.get('delta_t', 1)
        #error threshold
        self.epsilon = additional_params.get('epsilon', 1e-4)
        
        ###Initialize arrays for storing computations
        #voltage measured from the device
        self.v_meas = deque([0]*2, maxlen=2)
        #reference model: low pass filtered voltage (slowest)
        self.v_ref = deque([0]*2, maxlen=2)
        #adaptive power injection/offset
        self.mu = deque([0]*2, maxlen=2)    
        
    def get_action(self, env):
        """See parent class."""
        T = self.delta_t
        if not hasattr(self, 'node_id'):
            self.node_id = env.k.device.get_node_connected_to(self.device_id)

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))

        #assume device is computing the measured voltage
        #get measured voltage from device
        self.v_meas = env.k.device.devices[self.device_id]['device'].low_pass_filter_v

        #compute reference voltage - slower voltage measurement dynamics
        vl = (T * self.lpf_ref * (self.v_meas[1] + self.v_meas[0]) -(T * self.lpf_ref - 2) * (self.v_meas[1])) / \
                (2 + T * self.lpf_ref)

        self.v_ref.append(vl)
        
        ###########################################################
        #Logic to change direction of direct power injection
        dir = -1 #direction of the offset
        
        if env.k.time > 100:

            #compute error between measured voltage and low pass filtered meas. voltage
            e0 = -np.asarray(self.v_ref) + np.asarray(self.v_meas)
            #deadband for error
            if(LA.norm(e0,np.inf) < self.epsilon):
                e0 = np.zeros(np.shape(e0))

            #dynamic feedback term
            mu_arg = np.sign(e0) * e0
            mu_k = self.mu[1] + T/2 * (-dir) * self.gamma * sum(mu_arg)
            self.mu.append(mu_k)

        else:
            mu_k = 0
        return None, {'v_offset': mu_k, 'pow_inject': mu_k}

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
