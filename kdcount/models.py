import numpy
import kdcount

__all__ = ['dataset', 'points', 'field']

class dataset(object):
    def __init__(self, pos, boxsize, extra):
        """ create a dataset object for points located at pos in a boxsize.
            points is of (Npoints, Ndim)
            boxsize will be broadcasted to the dimension of points. 
            extra can be accessed as dataset.extra.
        """
        self.pos = pos
        self.tree = kdcount.build(self.pos, boxsize=boxsize)
        self.extra = extra

    def __len__(self):
        return len(self.pos)

class points(dataset):
    def __init__(self, pos, weight=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, boxsize, extra)
        self._weight = weight
        if weight is not None:
            assert len(weight.shape) == 1
            self.norm = weight.sum(axis=0)
        else:
            self.norm = len(pos) * 1.0
        self.subshape = ()

    def w(self, index):
        if self._weight is None:
            return 1.0
        else:
            return self._weight[index]

class field(dataset):
    def __init__(self, pos, value, weight=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, boxsize, extra)
        self._weight = weight
        if weight is not None:
            self._value = value * weight
        else:
            self._value = value
        self.subshape = value.shape[1:]
    def wv(self, index):
        return self._value[index]
    def w(self, index):
        if self._weight is None:
            return 1.0
        else:
            return self._weight[index]

