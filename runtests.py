import sys
import os

from sys import argv

from numpy.testing import Tester

if '-s' in sys.argv:
    # -s for in tree building
    sys.path.insert(0, os.path.abspath('.'))
    print("Testing in tree")
    argv.remove('-s')

tester = Tester()
r = tester.test(extra_argv=['-w', 'tests'] + argv[1:])
if not r:
    raise Exception("Tests failed")
