from kdcount import sphere
from kdcount import cluster
from kdcount import correlate
import numpy
from numpy.testing import assert_equal, assert_allclose


def test_cluster():
    numpy.random.seed(1234)
    dec = numpy.arcsin(numpy.random.uniform(-1, 1, size=10000)) / numpy.pi * 180
    ra = numpy.random.uniform(0, 2 * numpy.pi, size=10000) / numpy.pi * 180

    dataset = sphere.points(ra, dec)

    r = cluster.fof(dataset, 0.00001, np=None)

    assert r.N == len(dataset)

    binning = sphere.AngularBinning(numpy.linspace(0, 1.0, 10))

    r = correlate.paircount(dataset, dataset, binning=binning, usefast=False)
    assert_allclose( 
    r.sum1, 
    0.5 * numpy.diff(-numpy.cos(binning.angular_edges * numpy.pi / 180)) * len(ra) ** 2,
    rtol=1e-2)
