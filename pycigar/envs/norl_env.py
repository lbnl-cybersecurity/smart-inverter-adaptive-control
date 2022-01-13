import numpy as np
from pycigar.core.kernel.kernel import Kernel
import traceback
import atexit

class NoRLEnv:
    def __init__(self, sim_params, simulator='opendss'):
        """Initialize the environment.

        Parameters
        ----------
        sim_params : dict
            A dictionary of simulation information. for example: /examples/rl_config_scenarios.yaml
        simulator : str
            The name of simulator we want to use, by default it is OpenDSS.
        """
        self.state = None
        self.simulator = simulator

        # initialize the kernel
        self.k = Kernel(simulator=self.simulator,
                        sim_params=sim_params)

        # start an instance of the simulator (ex. OpenDSS)
        kernel_api = self.k.simulation.start_simulation()
        # pass the API to all sub-kernels
        self.k.pass_api(kernel_api)
        # start the corresponding scenario
        # self.k.scenario.start_scenario()

        # when exit the environment, trigger function terminate to clear all attached processes.
        atexit.register(self.terminate)


    def step(self):
        """See parent class.
        """

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            self.env_time += 1

            # perform action update for PV inverter device
            if len(self.k.device.get_pv_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_pv_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_pv_device_ids(), control_setting)


            if self.k.time <= self.k.t:
                self.k.update(reset=False)

                # check whether the simulator sucessfully solved the powerflow
                converged = self.k.simulation.check_converged()
                if not converged:
                    break

            if self.k.time >= self.k.t:
                break

        # the episode will be finished if it is not converged.
        done = not converged or (self.k.time == self.k.t)

        return done

    def reset(self):
        self.env_time = 0
        self.k.update(reset=True)
        self.sim_params = self.k.sim_params

        self.INIT_ACTION = {}
        pv_device_ids = self.k.device.get_pv_device_ids()
        for device_id in pv_device_ids:
            self.INIT_ACTION[device_id] = np.array(self.k.device.get_control_setting(device_id))

    def terminate(self):
        try:
            # close everything within the kernel
            self.k.close()
        except FileNotFoundError:
            print(traceback.format_exc())
