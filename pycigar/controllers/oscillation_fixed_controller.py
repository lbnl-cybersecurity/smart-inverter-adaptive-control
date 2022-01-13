import numpy as np
from pycigar.controllers.base_controller import BaseController


class OscillationFixedController(BaseController):
    """When this controller is triggered, the VV-VW breakpoints
    collapse around the average voltage at local node in the latest 10 timesteps.

    The goal of this controller is to create voltage oscillation.

    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params):
        """Instantiate."""
        BaseController.__init__(
            self,
            device_id
        )
        self.additional_params = additional_params
        self.trigger = False
        self.hack_curve = np.array([-0.001, 0.000, 0.000, 0.001, 0.002])
        self.average_span = 10

        self.action = None

    def get_action(self, env):
        """See parent class."""
        # calculate the average voltage of the last 10 timesteps
        # then collapse the curve around that point
        if self.trigger is False:
            node_id = env.k.device.devices[self.device_id]['node_id']
            if env.k.time - self.average_span - 1 > 0:
                vk = np.mean(np.abs(env.k.node.nodes[node_id]['voltage'][env.k.time - self.average_span - 1:env.k.time - 1]))
            else:
                vk = np.mean(np.abs(env.k.node.nodes[node_id]['voltage'][0:env.k.time - 1]))
            self.action = vk + self.hack_curve
            self.trigger = True
            return self.action
        else:
            return self.action

    def reset(self):
        """See parent class."""
        self.trigger = False
