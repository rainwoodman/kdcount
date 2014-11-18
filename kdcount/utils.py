from heapq import heappush, heappop

try:
    from sharedmem import MapReduce
except ImportError:
    class MapReduce(object):
        def __init__(self, np=None):
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
def divide_and_conquer(tree1, tree2, chunksize):
    """ lets try to always divide the smaller tree """
    def e(tree1, tree2):
        return -min(tree1.size, tree2.size), tree1, tree2
    heap = []
    heappush(heap, e(tree1, tree2))
    while True:
        w, x, y = heappop(heap)
        if w == 0: 
            heappush(heap, (0, x, y))
            break
        if x.less is None or y.less is None \
        or (x.size < chunksize or y.size < chunksize):
            heappush(heap, (0, x, y))
            continue
        if x.size < y.size:
            heappush(heap, e(x.less, y))
            heappush(heap, e(x.greater, y))
        else:
            heappush(heap, e(x, y.less))
            heappush(heap, e(x, y.greater))
    for w, x, y in heap:
        yield x, y
