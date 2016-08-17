from kdcount import models
import numpy
from numpy.testing import assert_equal, assert_array_equal

def test_indexing():

    pos = numpy.linspace(0, 1, 10000, endpoint=False).reshape(-1, 1)
    dataset = models.dataset(pos, boxsize=1.0)

    assert len(dataset[:10]) == 10
    assert_array_equal(dataset[:10].pos , pos[:10])
