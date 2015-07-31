import sys
import os

from sys import argv

from numpy.testing import Tester

tester = Tester()
r = tester.test(extra_argv=['-w', 'tests'] + argv[1:])
if r.failures or r.errors:
    raise Exception("Tests failed")
