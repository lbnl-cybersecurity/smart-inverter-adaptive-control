import numpy as np
from pycigar.controllers.base_controller import BaseController


class ImbalanceFixedController(BaseController):
    """When this controller is triggered, the VV-VW breakpoints
    will shift to the left or shift to the right according to
    the phase (a, b, c) of the local node.

    The goal of this controller is to create voltage imbalance.

    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params):
        """Instantiate an fixed Controller."""
        BaseController.__init__(
            self,
            device_id
        )
        self.additional_params = additional_params
        self.trigger = False
        self.hack_curve_all = np.array([1.01, 1.04, 1.04, 1.07, 1.09])
        self.hack_curve_a = np.array([1.01, 1.04, 1.04, 1.07, 1.09]) + 0.1
        self.hack_curve_b = np.array([0.95, 0.98, 0.98, 1.01, 1.04])
        self.hack_curve_c = np.array([0.95, 0.98, 0.98, 1.01, 1.04]) - 0.1
        self.action = None

    def get_action(self, env):
        """See parent class."""
        # change the VV-VW breakpoints according to the phase of local node.
        if self.trigger is False:
            if self.device_id[-1].isdigit():
                self.action = self.hack_curve_all
            elif self.device_id[-1] == 'a':
                self.action = self.hack_curve_a
            elif self.device_id[-1] == 'b':
                self.action = self.hack_curve_b
            elif self.device_id[-1] == 'c':
                self.action = self.hack_curve_c

            self.trigger = True
            return self.action
        else:
            return self.action

    def reset(self):
        """See parent class."""
        self.trigger = False
