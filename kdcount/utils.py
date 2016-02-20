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

from heapq import heappush, heappop

def toforest(root, chunksize):
    """ Divide a tree branch to a forest, 
        each subtree of size at most chunksize """
    heap = []
    heappush(heap, (-root.size, root))
    while True:
        w, x = heappop(heap)
        if w == 0: 
            heappush(heap, (0, x))
            break
        if x.less is None \
        or (x.size < chunksize):
            heappush(heap, (0, x))
            continue
        heappush(heap, (x.less.size, x.less))
        heappush(heap, (x.greater.size, x.greater))
    for w, x in heap:
        yield x
