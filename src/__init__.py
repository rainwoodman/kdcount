import inspect
import sys
import numpy
import pyximport
import os.path
pyximport.install(setup_args={'include_dirs': [numpy.get_include(), 
    os.path.dirname(inspect.getfile(sys.modules[__name__]))
    ]})

from pykdcount import *

from heapq import heappush, heappop

def divide_and_conquer(tree1, tree2, chunksize):
    def e(tree1, tree2):
        return -max(tree1.size, tree2.size), tree1, tree2
    heap = []
    heappush(heap, e(tree1, tree2))
    while True:
        w, x, y = heappop(heap)
        if w == 0: 
            heappush(heap, (0, x, y))
            break
        if x.less is None or y.less is None \
        or (x.size < chunksize and y.size < chunksize):
            heappush(heap, (0, x, y))
            continue
        if x.size > y.size:
            heappush(heap, e(x.less, y))
            heappush(heap, e(x.greater, y))
        else:
            heappush(heap, e(x, y.less))
            heappush(heap, e(x, y.greater))
    for w, x, y in heap:
        yield x, y
