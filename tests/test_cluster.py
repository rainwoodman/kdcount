from kdcount import cluster
import numpy
from numpy.testing import assert_equal

def test_cluster():

    pos = numpy.linspace(0, 1, 100, endpoint=False).reshape(-1, 1)
    dataset = cluster.dataset(pos, boxsize=1.0)

    r = cluster.fof(dataset, 0.008, np=None)
    assert r.N == 100
    assert (r.sum() == 1).all()
    r = cluster.fof(dataset, 0.011, np=None)
    assert r.N == 1
    assert (r.sum() == 100).all()

def test_parallel():
    pos = numpy.random.uniform(size=(10000, 2))
    dataset = cluster.dataset(pos, boxsize=1.0)
    r1 = cluster.fof(dataset, 0.01)
    r2 = cluster.fof(dataset, 0.01, np=4)

    assert r2.N == r1.N
    assert_equal(r2.sum(), r1.sum())

