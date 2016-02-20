from kdcount import correlate
import numpy

from numpy.testing import assert_almost_equal, assert_allclose

def test_simple():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(10, 3))
    dataset = correlate.points(pos, boxsize=1.0)
    binning = correlate.RBinning(numpy.linspace(0.5, 10))
    r = correlate.paircount(dataset, dataset, binning, np=0)

    r1 = correlate.paircount(dataset, dataset, binning, usefast=True, np=0)
    assert_allclose(
        r.sum1,
        r1.sum1)

def test_unweighted():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(4000, 3))
    dataset = correlate.points(pos, boxsize=1.0)
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))
    r = correlate.paircount(dataset, dataset, binning, np=0)

    assert_allclose(
        r.sum1,
        4 * numpy.pi / 3 * numpy.diff(r.edges ** 3) * len(dataset) ** 2,
        rtol=1e-2)
    r1 = correlate.paircount(dataset, dataset, binning, usefast=True, np=0)
    assert_allclose(
        r.sum1,
        r1.sum1)


def test_weighted():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(4000, 3))
    dataset = correlate.points(pos, boxsize=1.0, weights=numpy.ones(len(pos)))
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))
    r = correlate.paircount(dataset, dataset, binning, np=0)

    assert_allclose(
        r.sum1,
        4 * numpy.pi / 3 * numpy.diff(r.edges ** 3) * len(dataset) ** 2,
        rtol=1e-2)


def test_field():
    numpy.random.seed(1234)
    pos = numpy.random.uniform(size=(4000, 3))
    dataset = correlate.field(pos, value=numpy.ones(len(pos)), 
            boxsize=1.0, weights=numpy.ones(len(pos)))
    binning = correlate.RBinning(numpy.linspace(0, 0.5, 10))
    r = correlate.paircount(dataset, dataset, binning, np=0)

    assert_allclose(
        r.sum1,
        4 * numpy.pi / 3 * numpy.diff(r.edges ** 3) * len(dataset) ** 2,
        rtol=1e-2)

    assert_allclose(
        r.sum2,
        4 * numpy.pi / 3 * numpy.diff(r.edges ** 3) * len(dataset) ** 2,
        rtol=1e-2)


