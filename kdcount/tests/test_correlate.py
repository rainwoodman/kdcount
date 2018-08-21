from kdcount import correlate
import numpy

from numpy.testing import assert_almost_equal, assert_allclose, assert_array_less, assert_equal

def test_simple():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(10, 3))
    dataset = correlate.points(pos, boxsize=1.0)

    # use the python point point counting
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))

    # use the C node node counting
    binning1 = correlate.FastRBinning(numpy.linspace(0, 0.5, 10))

    r = correlate.paircount(dataset, dataset, binning, np=0)
    r1 = correlate.paircount(dataset, dataset, binning1, np=0)
    assert_equal( r.sum1, r1.sum1)

def test_unweighted():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(1000, 3))
    pos1 = pos[:, None, :]
    pos2 = pos[None, :, :]
    dist = pos1 - pos2
    dist[dist > 0.5] -= 1.0
    dist[dist < -0.5] += 1.0
    dist = numpy.einsum('ijk,ijk->ij', dist, dist) ** 0.5

    dataset = correlate.points(pos, boxsize=1.0)

    # use the python point point counting
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))

    # use the C node node counting
    binning1 = correlate.FastRBinning(numpy.linspace(0, 0.5, 10))


    dig = binning.edges.searchsorted(dist.flat, side='left')
    truth = numpy.bincount(dig)

    r = correlate.paircount(dataset, dataset, binning,  np=0)
    assert_equal( r.sum1, truth[1:-1])

    r1 = correlate.paircount(dataset, dataset, binning1,  np=0)
    assert_equal(r1.sum1, truth[1:-1])

def test_cross():
    numpy.random.seed(1234)
    pos1 = numpy.random.uniform(size=(10000, 2))
    pos2 = numpy.random.uniform(size=(10000, 2)) * 0.3
    dataset1 = correlate.points(pos1, boxsize=None)
    dataset2 = correlate.points(pos2, boxsize=None)

    # use the python point point counting
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))

    # use the C node node counting
    binning1 = correlate.FastRBinning(numpy.linspace(0, 0.5, 10))

    r1 = correlate.paircount(dataset1, dataset2, binning, np=0)
    r2 = correlate.paircount(dataset1, dataset2, binning1, np=0)
    assert_equal(r1.sum1, r2.sum1)
    r3 = correlate.paircount(dataset1, dataset2, binning, np=4)
    assert_equal(r1.sum1, r3.sum1)
    r4 = correlate.paircount(dataset1, dataset2, binning1, np=4)
    assert_equal(r1.sum1, r4.sum1)

def test_weighted():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(1000, 3))
    datasetw = correlate.points(pos, boxsize=1.0, weights=numpy.ones(len(pos)))
    dataset = correlate.points(pos, boxsize=1.0)
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))
    r = correlate.paircount(datasetw, datasetw, binning, np=0)
    r1 = correlate.paircount(dataset, dataset, binning, np=0)

    assert_equal( r.sum1, r1.sum1)


def test_field():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(1000, 3))
    dataset = correlate.field(pos, value=numpy.ones(len(pos)), 
            boxsize=1.0, weights=numpy.ones(len(pos)))
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))
    r = correlate.paircount(dataset, dataset, binning, np=0)

    assert_allclose(r.sum1, r.sum2)

def test_channels():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(1000, 3))
    datasetw = correlate.points(pos, boxsize=1.0, weights=numpy.ones(len(pos)))
    dataset = correlate.points(pos, boxsize=1.0)

    binning_mc1 = correlate.FlatSkyMultipoleBinning(numpy.linspace(0, 0.5, 10), ells=[0, 0, 0], los=0)
    binning_mc2 = correlate.MultipoleBinning(numpy.linspace(0, 0.5, 10), ells=[0, 0, 0])
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))

    r_mc1 = correlate.paircount(datasetw, datasetw, binning_mc1, np=0)
    r_mc2 = correlate.paircount(datasetw, datasetw, binning_mc2, np=0)
    r1 = correlate.paircount(dataset, dataset, binning, np=0)

    assert_equal( r_mc1.sum1[0], r1.sum1)
    assert_equal( r_mc2.sum1[0], r1.sum1)
