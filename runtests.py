import sys
import os

from sys import argv

from numpy.testing import Tester

sys.path.insert(0, 'kdcount')

tester = Tester()
if len(argv) == 1:
    r = tester.test(extra_argv=['-w', 'tests'] + argv[1:])
else:
    r = tester.test(extra_argv=argv[1:])
if r.failures or r.errors:
    raise Exception("Tests failed")
