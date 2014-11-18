import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kdcount import cluster
import numpy

def test_p():
    pos = numpy.load(os.path.join(os.path.dirname(__file__),
        'TEST-A00_hodfit-small.npy'))

    dataset = cluster.dataset(pos, boxsize=1.0)
    r = cluster.fof(dataset, 0.02)
    print len(pos), r.N, r.labels 
test_p()
