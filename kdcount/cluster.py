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
        while self.once(shmnext, np) > 0:
            pass
        u, labels = numpy.unique(self.next, return_inverse=True)
        self.N = len(u)
        self.labels = labels
        
    def once(self, shmnext, np):
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
                    with pool.critical:
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

