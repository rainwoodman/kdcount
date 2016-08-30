import kdcount
import numpy

from kdcount import correlate
from kdcount import bootstrap
from numpy.testing import assert_allclose, assert_array_equal

def test_onepoint():
    numpy.random.seed(1234)
    data = (-numpy.arange(20)) % 10
    bsfun = lambda x: x
    policy = bootstrap.policy(bsfun, data)
    estimator = lambda x : sum(x)
    result = policy.run(estimator, data)
    for i in range(10):
        L, R = policy.resample(result, [i])
        assert_array_equal(2 * i, R)
        assert_array_equal(2, L)

def test_twopoint():
    numpy.random.seed(1234)
    data = (-numpy.arange(20)) % 10
    bsfun = lambda x: x
    policy = bootstrap.policy(bsfun, data)
    estimator = lambda x, y : sum(x) * sum(y)
    result = policy.run(estimator, data, data)
    for i in range(10):
        L, R = policy.resample(result, [i])
        assert_array_equal((2 * i) ** 2, R)
        assert_array_equal((2, 2), L)

    for i in range(9):
        L, R = policy.resample(result, [i, i + 1])
        assert_array_equal((2 * i) * (2 * (i + 1)) * 2 + (2 * i) ** 2 + (2 * (i + 1)) ** 2, R)
        assert_array_equal((4, 4), L)

def test_paircount():
    numpy.random.seed(1234)
    data = 1.0 * ((-numpy.arange(4).reshape(-1, 1)) % 2)
    data = correlate.points(data)
    bsfun = lambda x: numpy.int32(x.pos[:, 0])
    policy = bootstrap.policy(bsfun, data)
    binning=correlate.RBinning(numpy.linspace(0, 100, 2, endpoint=True))

    def estimator( x, y):
        r = correlate.paircount(x, y, binning, usefast=False, np=0)
        return r.fullsum1
    result = policy.run(estimator, data, data)
    L, R = policy.resample(result, numpy.arange(2))

    assert_array_equal(L, (4, 4))
    assert_array_equal(R, (8, 8, 0))
