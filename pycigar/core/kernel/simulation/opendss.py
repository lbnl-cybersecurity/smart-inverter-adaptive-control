from pycigar.core.kernel.simulation import KernelSimulation
from pycigar.utils.opendss.pseudo_api import PyCIGAROpenDSSAPI

RETRIES_ON_ERROR = 10


class OpenDSSSimulation(KernelSimulation):

    def __init__(self, master_kernel):
        """See parent class."""
        KernelSimulation.__init__(self, master_kernel)
        self.opendss_proc = None

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_simulation(self):
        """See parent class."""
        return PyCIGAROpenDSSAPI()

    def update(self, reset):
        """See parent class."""
        self.kernel_api.simulation_step()

    def close(self):
        """See parent class."""
        # self.teardown_opendss()
        pass

    def check_converged(self):
        """See parent class."""
        return self.kernel_api.check_simulation_converged()

