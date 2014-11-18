import numpy
from models import dataset
import utils
class fof(object):
    def __init__(self, data, linking_length, np=None):
        self.data = data
        self.linking_length = linking_length

        # iterations
        shmhead = utils.empty(len(data), dtype='intp')
        self.iterations = 0
        head = numpy.arange(len(data), dtype='intp')
        op = 1 
        while op > 0:
            op = self._once(shmhead, head, np)
            self.iterations = self.iterations + 1
        u, labels = numpy.unique(head, return_inverse=True)
        self.N = len(u)
        length = utils.bincount(labels, 1, self.N)
        # for example old labels == 5 is the longest halo
        # then a[0] == 5
        # we want to replace in labels 5 to 0
        # thus we need an array inv[5] == 0
        a = length.argsort()[::-1]
        length = length[a]
        inv = numpy.empty(self.N, dtype='intp')
        inv[a] = numpy.arange(self.N)
        #print inv.max(), inv.min()
        self.labels = inv[labels]
        self.length = length
        self.offset = numpy.empty_like(length)
        self.offset[0] = 0
        self.offset[1:] = length.cumsum()[:-1]
        self.indices = self.labels.argsort() 

    def find(self, groupid):
        """ return all of the indices of particles of groupid """
        return self.indices[self.offset[groupid]
                :self.offset[groupid]+ self.length[groupid]]

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

    def _once(self, shmhead, head, np):
        """ fof iteration,
            shmhead is the output and shall be write-shared by all ranks
            head is the input can be readonly-shared."""
        tree = self.data.tree
        if np != 0:
            p = list(utils.divide_and_conquer(tree, tree, 10000))
        else:
            p = [(tree, tree)]

        #print 'p', len(p)
        ll = self.linking_length
        ops = [0]
        shmhead[:] = head

        with utils.MapReduce(np=np) as pool:
            def work(iwork):
                n1, n2 = p[iwork]
                operations = [0]
                def process(r, i, j):
                    if len(r) == 0: return 
#                    print iwork, 'len(r)', len(r)
        #            with pool.critical:

                    # update the head id; 
                    # only for those that would decrease
                    mask2 = shmhead[i] > head[j]
                    i = i[mask2]
                    j = j[mask2]
                    nj = head[j]
                    # find the minimal id for each particle
                    # to be updated
                    #  sort by i and -nj
                    arg = numpy.lexsort((i, -nj))
                    i = i[arg]
                    nj = nj[arg]
                    #  find the last item in each i
                    lasts = (i[1:] != i[:-1]).nonzero()[0]
                    i = i[lasts]
                    nj = nj[lasts]

                    #  write to each particle, once, in order
                    #  minimizing memory clashes from many ranks;
                    #  the algorithm is stable against racing
                    #  but it would slow down the cache if each rank
                    #  were directly writing.
                    shmhead[i] = nj
                    operations[0] += len(i)
#                    print iwork, 'len(r)', len(i)
                n1.enum(n2, ll, process, bunch=10000 * 8)
#                print 'work', iwork, 'done'
                return operations[0]
            def reduce(op):
                #print ops[0]
                ops[0] += op

            pool.map(work, range(len(p)), reduce=reduce)

        head[:] = shmhead
        return ops[0]

