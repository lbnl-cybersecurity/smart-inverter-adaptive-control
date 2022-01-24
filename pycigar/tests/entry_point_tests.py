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
