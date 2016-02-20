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

