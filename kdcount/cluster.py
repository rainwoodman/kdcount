import numpy
from models import dataset
import utils
class fof(object):
    def __init__(self, data, linking_length, np=None):
        self.data = data
        self.linking_length = linking_length
        self.next = numpy.arange(len(data), dtype='intp')

        # iterations
        shmnext = utils.empty(len(data), dtype='intp')
        self.iterations = 0
        N = 1 
        while N > 0:
            N = self._once(shmnext, np)
            self.iterations = self.iterations + 1
            #print N
        u, labels = numpy.unique(self.next, return_inverse=True)
        self.N = len(u)
        self.labels = labels

    def sum(self, weights=None):
        """ return the sum of weights of each object """
        if weights is None:
            weights = self.data._weights
        if weights is None:
            weights = 1.0
        return utils.bincount(self.labels, weights, self.N)

    def center(self, weights=None):
        """ return the center of each object """
        if weights is None:
            weights = self.data._weights
        if weights is None:
            weights = 1.0
        mass = utils.bincount(self.labels, weights, self.N)
        cp = numpy.empty((len(mass), self.data.pos.shape[-1]), 'f8')
        for d in range(self.data.pos.shape[-1]):
            cp[..., d] = utils.bincount(self.labels, weights *
                    self.data.pos[..., d], self.N)
            cp[..., d] /= mass
        return cp

    def _once(self, shmnext, np):
        tree = self.data.tree
        if np != 0:
            p = list(utils.divide_and_conquer(tree, tree, 10000))
        else:
            p = [(tree, tree)]

        ll = self.linking_length
        ops = [0]
        next = self.next
        shmnext[:] = self.next

        with utils.MapReduce(np=np) as pool:
            def work(i):
                n1, n2 = p[i]
                operations = 0
                for r, i, j in n1.enumiter(n2, ll):
                    if len(r) == 0: continue
        #            with pool.critical:
                    if True:
                        mask2 = shmnext[i] > next[j]
                        i = i[mask2]
                        j = j[mask2]
                        shmnext[i] = next[j]
                        operations += mask2.sum()
                return operations 
            def reduce(op):
                ops[0] += op

            pool.map(work, range(len(p)), reduce=reduce)

        next[:] = shmnext
        return ops[0]

