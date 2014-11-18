import numpy
import kdcount

__all__ = ['dataset', 'points', 'field']

class dataset(object):
    def __init__(self, pos, weights=None, boxsize=None, extra={}):
        """ create a dataset object for points located at pos in a boxsize.
            points is of (Npoints, Ndim)
            boxsize will be broadcasted to the dimension of points. 
            extra can be accessed as dataset.extra.
        """
        self.pos = pos
        self.boxsize = boxsize
        self.tree = kdcount.build(self.pos, boxsize=boxsize)
        self.extra = extra
        self._weights = weights

    def w(self, index):
        """ weight at index """
        if self._weights is None:
            return 1.0
        else:
            return self._weights[index]

    def __len__(self):
        return len(self.pos)

class points(dataset):
    def __init__(self, pos, weights=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, weights, boxsize, extra)
        if weights is not None:
            assert len(weights.shape) == 1
            self.norm = weights.sum(axis=0)
        else:
            self.norm = len(pos) * 1.0
        self.subshape = ()

class field(dataset):
    def __init__(self, pos, value, weights=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, weights, boxsize, extra)
        if weights is not None:
            self._value = value * weights
        else:
            self._value = value
        self.subshape = value.shape[1:]
    def wv(self, index):
        """ weight * value """
        return self._value[index]
