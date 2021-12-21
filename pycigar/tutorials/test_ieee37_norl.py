"""import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.controllers.custom_adaptive_inverter_controller import CustomAdaptiveInverterController
from pycigar.utils.registry import register_devcon

misc_inputs = pycigar.DATA_DIR + '/ieee37busdata/misc_inputs.csv'
dss = pycigar.DATA_DIR + '/ieee37busdata/ieee37.dss'
load_solar = pycigar.DATA_DIR + '/ieee37busdata/load_solar_data.csv'
breakpoints = pycigar.DATA_DIR + '/ieee37busdata/breakpoints.csv'

sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=False)
start = 100
sim_params['scenario_config']['start_end_time'] = [start, start + 750]  # fix the exp start and end time
sim_params['env_config']['sims_per_step'] = 1 # on 1 step call, it is equal to 1 simulation step.
del sim_params['attack_randomization'] # turn off the attack randomization

register_devcon('custom_adaptive_inverter_controller', CustomAdaptiveInverterController)
from pycigar.envs.norl_env import NoRLEnv
env = NoRLEnv(sim_params=sim_params) # init env with the sim_params above

env.reset()
done = False
while not done:
    done = env.step() # every step call will return done, the status whether the exp is finished or not"""

import pycigar
import pandas as pd
from pycigar.utils.input_parser import input_parser

from pycigar.utils.registry import register_devcon
from pycigar.controllers.custom_adaptive_inverter_controller import CustomAdaptiveInverterController
from pycigar.controllers.adaptive_inverter_controller import AdaptiveInverterController
from pycigar.controllers.custom_hack_controller import CustomHackController
from pycigar.controllers.fixed_controller import FixedController

register_devcon('Attacker', CustomHackController)
register_devcon('Defender', CustomAdaptiveInverterController)

file_misc_inputs_path = pycigar.DATA_DIR + '/ieee37busdata/misc_inputs.csv'