import numpy
from heapq import heappush, heappop

from . import pykdcount as _core

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
    __nodeclass__ = KDNode
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

