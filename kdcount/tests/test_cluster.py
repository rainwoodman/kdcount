from kdcount import cluster
import numpy
from numpy.testing import assert_equal, run_module_suite

def test_cluster():

    pos = numpy.linspace(0, 1, 10000, endpoint=False).reshape(-1, 1)
    dataset = cluster.dataset(pos, boxsize=1.0)

    r = cluster.fof(dataset, 1.1 / len(pos))
    assert r.N == 1
    assert (r.sum() == len(pos)).all()

    r = cluster.fof(dataset, 0.8 / len(pos))
    assert r.N == len(pos)
    assert (r.sum() == 1).all()


