import os
import pandas as pd
import numpy as np


def input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path=None, benchmark=False, percentage_hack=0.40, vectorized_mode=False):
    """Take multiple .csv files and parse them into the .yml file that required by pycigar.
    Parameters
    ----------
    misc_inputs_path : str
        directory to miscellaneous settings for the experiment. Example can be found at ./data/ieee37busdata/misc_inputs.csv
    dss_path : str
        directory to .dss file. Example can be found at ./data/ieee37busdata/ieee37.dss
    load_solar_path : str
        directory to load and solar profiles for different inverters. Example can be found at ./data/ieee37busdata/load_solar_data.csv
    breakpoints_path : str, optional
        directory to default settings of different inverters. Defaults to None to use the default settings in this function, by default None
    benchmark : bool, optional
        whether the experiment is in benchmark mode. If true, disable the randomization at inverters filter. Defaults to False, by default False
    percentage_hack : float, optional
        percentage hack for all devices. Defaults to 0.45. Only have meaning when benchmark is True, by default 0.45
    adv : bool, optional
        whether the experiment is adversarial training. Defaults to False. If True, set the advesarial devices to use RL controllers, by default False
    Returns
    -------
    dict
        a dictionary contains full information to run the experiment
    """

    file_misc_inputs_path = misc_inputs_path
    file_dss_path = dss_path
    file_load_solar_path = load_solar_path
    file_breakpoints_path = breakpoints_path

    json_query = {
        'tune_search': True,
        'vectorized_mode': False,

        'hack_setting': {'default_control_setting': [1.039, 1.04, 1.04, 1.041, 1.042]},

        'env_config': {
            'clip_actions': True,
            'sims_per_step': 35
        },
        'attack_randomization': {
            'generator': 'AttackDefinitionGenerator'
        },
        'simulation_config': {
            'network_model_directory': file_dss_path,
            'custom_configs': {'solution_mode': 1,
                              'solution_number': 1,
                              'solution_step_size': 1,
                              'solution_control_mode': 2,
                              'solution_max_control_iterations': 1000000,
                              'solution_max_iterations': 30000,
                              'power_factor': 0.9},
        },
        'scenario_config': {
            'multi_config': True,
            'start_end_time': 750,
            'network_data_directory': file_load_solar_path,
            'custom_configs': {'load_scaling_factor': 1.5,
                             'solar_scaling_factor': 3,
                             'slack_bus_voltage': 1.02,  # default 1.04
                             'load_generation_noise': False,
                             'power_factor': 0.9},
            'nodes': [],
            'regulators': {
                'max_tap_change': 30,
                'forward_band': 16,
                'tap_number': 2,
                'tap_delay': 0,
                'delay': 30
            }
        }
    }

    # read misc_input
    misc_inputs_data = pd.read_csv(file_misc_inputs_path, index_col=0, names=['parameter', 'value'])
    misc_inputs_data.value = pd.to_numeric(misc_inputs_data.value, downcast='float')

    power_factor = misc_inputs_data.value['power factor']
    load_scaling_factor = misc_inputs_data.value['load scaling factor']
    solar_scaling_factor = misc_inputs_data.value['solar scaling factor']

    json_query['vectorized_mode'] = vectorized_mode
    json_query['scenario_config']['custom_configs']['load_scaling_factor'] = load_scaling_factor
    json_query['scenario_config']['custom_configs']['solar_scaling_factor'] = solar_scaling_factor
    json_query['scenario_config']['custom_configs']['power_factor'] = power_factor

    low_pass_filter_measure_mean = misc_inputs_data.value['measurement filter time constant mean']
    low_pass_filter_measure_std = misc_inputs_data.value['measurement filter time constant std']
    low_pass_filter_output_mean = misc_inputs_data.value['output filter time constant mean']
    low_pass_filter_output_std = misc_inputs_data.value['output filter time constant std']
    hack_start = misc_inputs_data.value['hack start']
    hack_end = misc_inputs_data.value['hack end']
    hack_update = misc_inputs_data.value['hack update']
    gamma = misc_inputs_data.value['gamma']
    k = misc_inputs_data.value['k']
    lpfm_avg = misc_inputs_data.value['measurement filter time constant avg v']

    default_control_setting = [misc_inputs_data.value['bp1 default'],
                               misc_inputs_data.value['bp2 default'],
                               misc_inputs_data.value['bp3 default'],
                               misc_inputs_data.value['bp4 default'],
                               misc_inputs_data.value['bp5 default']]

    # read load_solar_data & read
    load_solar_data = pd.read_csv(file_load_solar_path)
    node_names = [node for node in list(load_solar_data) if '_pv' not in node]
    if file_breakpoints_path is not None:
        breakpoints_data = pd.read_csv(file_breakpoints_path)
    else:
        breakpoints_data = None

    for node in node_names:
        node_default_control_setting = default_control_setting
        if breakpoints_data is not None and node + '_pv' in list(breakpoints_data):
            node_default_control_setting = breakpoints_data[node + '_pv'].tolist()

        node_description = {}
        node_description['name'] = node.lower()
        node_description['load_profile'] = None
        node_description['devices'] = []
        device = {}
        device['name'] = 'inverter_' + node.lower()

        # configuration for device
        device['device'] = 'custom_pv_device'
        device['custom_device_configs'] = {}
        device['custom_device_configs']['default_control_setting'] = node_default_control_setting
        device['custom_device_configs']['delay_timer'] = 60
        device['custom_device_configs']['threshold'] = 0.05
        device['custom_device_configs']['adaptive_gain'] = 1e5
        device['custom_device_configs']['is_butterworth_filter'] = False
        device['custom_device_configs']['k'] = k
        device['custom_device_configs']['gamma'] = gamma
        device['custom_device_configs']['lpf_m'] = low_pass_filter_measure_mean
        device['custom_device_configs']['lpf_avg'] = lpfm_avg
        device['custom_device_configs']['low_pass_filter_measure_mean'] = low_pass_filter_measure_mean
        device['custom_device_configs']['low_pass_filter_output_mean'] = low_pass_filter_output_mean
        if not benchmark:
            device['custom_device_configs']['low_pass_filter_measure_std'] = low_pass_filter_measure_std
            device['custom_device_configs']['low_pass_filter_output_std'] = low_pass_filter_output_std

        # configuration of controller
        device['controller'] = 'Defender'
        device['custom_controller_configs'] = {}
        device['custom_controller_configs']['default_control_setting'] = node_default_control_setting
        device['custom_controller_configs']['adaptive_gain'] = 1e5
        device['custom_controller_configs']['gamma'] = gamma
        device['custom_controller_configs']['k'] = k
        device['custom_controller_configs']['lpf_m'] = low_pass_filter_measure_mean
        device['custom_controller_configs']['lqf_avg'] = lpfm_avg

        # configuration of adversarial controller
        device['adversary_controller'] = 'Attacker'
        device['adversary_custom_controller_configs'] = {}
        device['adversary_custom_controller_configs']['default_control_setting'] = [1.014, 1.015, 1.015, 1.016, 1.017]
        device['adversary_custom_controller_configs']['y_threshold'] = 0.03
        device['adversary_custom_controller_configs']['hack_update'] = hack_update
        device['hack'] = [hack_start, percentage_hack, hack_end]
        node_description['devices'].append(device)

        json_query['scenario_config']['nodes'].append(node_description)

    max_tap_change = misc_inputs_data.value['max tap change default']
    forward_band = misc_inputs_data.value['forward band default']
    tap_number = misc_inputs_data.value['tap number default']
    tap_delay = misc_inputs_data.value['tap delay default']

    json_query['scenario_config']['regulators']['max_tap_change'] = max_tap_change
    json_query['scenario_config']['regulators']['forward_band'] = forward_band
    json_query['scenario_config']['regulators']['tap_number'] = tap_number
    json_query['scenario_config']['regulators']['tap_delay'] = tap_delay

    return json_query
