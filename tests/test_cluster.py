from kdcount import cluster
import numpy
from numpy.testing import assert_equal, run_module_suite

def test_cluster():

    pos = numpy.linspace(0, 1, 10000, endpoint=False).reshape(-1, 1)
    dataset = cluster.dataset(pos, boxsize=1.0)

    r = cluster.fof(dataset, 1.1 / len(pos), np=0, verbose=True)
    print r.N
    assert r.N == 1
    assert (r.sum() == len(pos)).all()

    r = cluster.fof(dataset, 0.8 / len(pos), np=0, verbose=True)
    assert r.N == len(pos)
    assert (r.sum() == 1).all()


def test_parallel():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(100000, 2))
    dataset = cluster.dataset(pos, boxsize=1.0)
    r1 = cluster.fof(dataset, 0.03, np=0, verbose=True)
    r2 = cluster.fof(dataset, 0.03, np=4, verbose=True)

    assert r2.N == r1.N
    print r2.length
    assert_equal(r2.sum(), r1.sum())

