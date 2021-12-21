import unittest
import sys
import pycigar

loader = unittest.TestLoader()
start_dir = pycigar.PROJECT_DIR + '/tests/fast_tests'
print(start_dir)
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
res = runner.run(suite)
if not res.wasSuccessful():
    sys.exit(1)

# Test no-agent path
pycigar.main(
    pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv",
    pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss",
    pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv",
    pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv",
    2,
    None,
    pycigar.LOG_DIR,
)

# TODO: clean up the API to handle optional arguments.
