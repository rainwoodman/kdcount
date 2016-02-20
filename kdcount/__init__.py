import numpy

import pykdcount as _core

class KDNode(_core.KDNode):
    def __repr__(self):
        return ('KDNode(dim=%d, split=%d, size=%d)' % 
                (self.dim, self.split, self.size))


class KDTree(_core.KDTree):
    __nodeclass__ = KDNode
    def __repr__(self):
        return ('KDTree(size=%d, thresh=%d, boxsize=%s, input=%s)' % 
            (self.size, self.thresh, str(self.boxsize), str(self.input.shape)))

class KDAttr(_core.KDAttr):
    def __repr__(self):
        return ('KDAttr(input=%s)' % (str(self.input.shape)))

