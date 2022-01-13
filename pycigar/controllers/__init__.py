from pycigar.controllers.base_controller import BaseController
from pycigar.controllers.fixed_controller import FixedController
from pycigar.controllers.mimic_controller import MimicController
from pycigar.controllers.oscillation_fixed_controller import OscillationFixedController
from pycigar.controllers.imbalance_fixed_controller import ImbalanceFixedController

from pycigar.utils.pycigar_registration import pycigar_register, pycigar_make, pycigar_spec

__all__ = [
    "BaseController",
    "FixedController",
    "MimicController",
    "OscillationFixedController",
    "ImbalanceFixedController"
]

pycigar_register(
    id='mimic_controller',
    entry_point='pycigar.controllers:MimicController'
)

pycigar_register(
    id='oscillation_fixed_controller',
    entry_point='pycigar.controllers:OscillationFixedController'
)

pycigar_register(
    id='imbalance_fixed_controller',
    entry_point='pycigar.controllers:ImbalanceFixedController'
)

pycigar_register(
    id='fixed_controller',
    entry_point='pycigar.controllers:FixedController'
)

