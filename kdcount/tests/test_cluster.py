from kdcount import cluster
import numpy
from numpy.testing import assert_array_equal, run_module_suite

def test_cluster():

    pos = numpy.linspace(0, 1, 10000, endpoint=False).reshape(-1, 1)
    dataset = cluster.dataset(pos, boxsize=1.0)

    r = cluster.fof(dataset, 0.8 / len(pos))
    assert_array_equal(r.N, len(pos))
    assert_array_equal(r.sum(), 1)

    r = cluster.fof(dataset, 1.1 / len(pos))

    assert_array_equal(r.N, 1)
    assert_array_equal(r.sum(), len(pos))


def test_cluster_empty():

    pos = numpy.empty((0, 3))
    dataset = cluster.dataset(pos, boxsize=1.0)

    # no error shall be raised
    r = cluster.fof(dataset, 0.8)



