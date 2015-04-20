import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kdcount import cluster
import numpy

def test_p(ll):
    pos = numpy.load(os.path.join(os.path.dirname(__file__),
#        'TEST-A00_hodfit-big.npy'
        'TEST-qpm_06352_0001-big.npy'
))

    print pos.max(axis=0), pos.min(axis=0)
    dataset = cluster.dataset(pos, boxsize=1.0)
    r = cluster.fof(dataset, ll, np=None)
    mass = r.sum()
    center = r.center()
    print 'linking length', ll
    print 'particles', len(pos)
    print 'groups', r.N
    print 'iterations', r.iterations
    print 'counts in mass', numpy.bincount(numpy.int32(mass))
    print 'center of most massive group', center[mass.argmax()]
    print 'center of first group', center[0]
    print 'center of first group check', \
        numpy.mean(pos[r.find(0)], axis=0, dtype='f8')
test_p(0.0001)
test_p(0.0004559498158012794)
test_p(0.001)
test_p(0.002)
test_p(0.005)
test_p(0.01)
test_p(0.02)
test_p(0.1)