from .version import __version__
import numpy
from heapq import heappush, heappop

from . import pykdcount as _core
from numpy.lib.stride_tricks import broadcast_arrays

class KDNode(_core.KDNode):
    def __repr__(self):
        return ('KDNode(dim=%d, split=%d, size=%d)' % 
                (self.dim, self.split, self.size))

    def enumiter(self, other, rmax, bunch=100000):
        """ cross correlate with other, for all pairs
            closer than rmax, iterate.

            for r, i, j in A.enumiter(...):
                ...
            where r is the distance, i and j are the original
            input array index of the data.

            This uses a thread to convert from KDNode.enum.
        """

        def feeder(process):
            self.enum(other, rmax, process, bunch)
        for r, i, j in makeiter(feeder):
            yield r, i, j

    def enum(self, other, rmax, process=None, bunch=100000, **kwargs):
        """ cross correlate with other, for all pairs
            closer than rmax, iterate.

            >>> def process(r, i, j, **kwargs):
            >>>    ...

            >>> A.enum(... process, **kwargs):
            >>>   ...

            where r is the distance, i and j are the original
            input array index of the data. arbitrary args can be passed
            to process via kwargs.
        """
        rall = None
        if process is None:
            rall = [numpy.empty(0, 'f8')]
            iall = [numpy.empty(0, 'intp')]
            jall = [numpy.empty(0, 'intp')]
            def process(r1, i1, j1, **kwargs):
                rall[0] = numpy.append(rall[0], r1)
                iall[0] = numpy.append(iall[0], i1)
                jall[0] = numpy.append(jall[0], j1)

        _core.KDNode.enum(self, other, rmax, process, bunch, **kwargs)

        if rall is not None:
            return rall[0], iall[0], jall[0]
        else:
            return None

    def count(self, other, r, attrs=None, info={}):
        """ Gray & Moore based fast dual tree counting.

            r is the edge of bins:

            -inf or r[i-1] < count[i] <= r[i]

            attrs: None or tuple
                if tuple, attrs = (attr_self, attr_other)

            Returns: count, 
                count, weight of attrs is not None
        """

        r = numpy.array(r, dtype='f8')

        return _core.KDNode.count(self, other, r, attrs, info=info)

    def fof(self, linkinglength, out=None):
        """ Friend-of-Friend clustering with linking length.

            Returns: the label
        """
        if out is None:
            out = numpy.empty(self.size, dtype='intp')
        return _core.KDNode.fof(self, linkinglength, out)

    def integrate(self, min, max, attr=None, info={}):
        """ Calculate the total number of points between [min, max).

            If attr is given, also calculate the sum of the weight.

            This is a M log(N) operation, where M is the number of min/max
            queries and N is number of points.

        """
        if numpy.isscalar(min):
            min = [min for i in range(self.ndims)]
        if numpy.isscalar(max):
            max = [max for i in range(self.ndims)]

        min = numpy.array(min, dtype='f8', order='C')
        max = numpy.array(max, dtype='f8', order='C')

        if (min).shape[-1] != self.ndims:
            raise ValueError("dimension of min does not match Node")
        if (max).shape[-1] != self.ndims:
            raise ValueError("dimension of max does not match Node")
        min, max = broadcast_arrays(min, max)
        return _core.KDNode.integrate(self, min, max, attr, info)

    def make_forest(self, chunksize):
        """ Divide a tree branch to a forest, 
            each subtree of size at most chunksize """
        heap = []
        heappush(heap, (-self.size, self))
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

class KDTree(_core.KDTree):
    """ KDTree

        KDTree.root is the root node. The algorithms are implemented
        as methods of the node.

        Parameters
        ----------
        input : array_like
            single or double array of shape (N, ndims).
        boxsize : array_like or scalar
            If given, the input data is on a torus with periodic boundry.
            the size of the torus is given by boxsize.
        thresh : int
            minimal size of a leaf.

    """
    __nodeclass__ = KDNode
    def __init__(self, input, boxsize=None, thresh=10):
        _core.KDTree.__init__(self, input, boxsize, thresh)

    def __repr__(self):
        return ('KDTree(size=%d, thresh=%d, boxsize=%s, input=%s)' % 
            (self.size, self.thresh, str(self.boxsize), str(self.input.shape)))

class KDAttr(_core.KDAttr):
    def __repr__(self):
        return ('KDAttr(input=%s)' % (str(self.input.shape)))

import threading
try:
    import Queue as queue
except ImportError:
    import queue
import signal
def makeiter(feeder):
    q = queue.Queue(2)
    def process(*args):
        q.put(args)
    def wrap(process):
        try:
            feeder(process)
        except Exception as e:
            q.put(e)
        finally:
            q.put(StopIteration)
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    t = threading.Thread(target=wrap, args=(process,))
    t.start()
    signal.signal(signal.SIGINT, old)
    while True:
        item = q.get()
        if item is StopIteration:
            q.task_done()
            q.join()
            t.join()
            break
        elif isinstance(item, Exception):
            q.task_done()
            q.join()
            t.join()
            raise item
        else:
            if len(item) == 1: item = item[0]
            yield item
            q.task_done()

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

