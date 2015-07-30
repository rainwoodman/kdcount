from kdcount import cluster
import numpy

def test_cluster():

    pos = numpy.linspace(0, 1, 100, endpoint=False).reshape(-1, 1)
    dataset = cluster.dataset(pos, boxsize=1.0)

    r = cluster.fof(dataset, 0.008, np=None)
    assert r.N == 100
    assert (r.sum() == 1).all()
    r = cluster.fof(dataset, 0.011, np=None)
    assert r.N == 1
    assert (r.sum() == 100).all()

