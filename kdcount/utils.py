import numpy

try:
    from sharedmem import MapReduce
    from sharedmem import empty
except ImportError:
    import numpy
    empty = numpy.empty
    class MapReduce(object):
        def __init__(self, np=None):
            self.critical = self
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, work, items, reduce=None):
            if reduce is not None: 
                callreduce = lambda r: \
                    reduce(*r) if isinstance(r, tuple) \
                        else reduce(r)
            else:
                callreduce = lambda r: r
            return [callreduce(work(i)) for i in items]

def bincount(dig, weight, minlength):
    """ bincount supporting scalar and vector weight """
    if numpy.isscalar(weight):
        return numpy.bincount(dig, minlength=minlength) * weight
    else:
        return numpy.bincount(dig, weight, minlength)

# for creating dummy '1.0' arrays
from numpy.lib.stride_tricks import as_strided

class constant_array(numpy.ndarray):
    def __new__(kls, shape, dtype='f8'):
        if numpy.isscalar(shape):
            shape = (shape,)
        foo = numpy.empty((), dtype=dtype)
        self = as_strided(foo, list(shape) + list(foo.shape), 
                [0] * len(shape) + list(foo.strides)).view(type=constant_array)
        self.value = foo
        return self

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end, step = key.indices(len(self))
            N = (end - start) // step
        elif isinstance(key, (list,)):
            N = len(key)
        elif isinstance(key, (numpy.ndarray,)):
            if key.dtype == numpy.dtype('?'):
                N = key.sum()
            else:
                N = len(key)
        else:
            N = None
        if N is None:
            return numpy.ndarray.__getitem__(self, key)
        else:
            shape = [N] + list(self.shape[1:])
            r = constant_array(shape, self.dtype)
            r.value[...] = self.value[...]
            return r

