import numpy
from models import dataset
import utils
class fof(object):
    def __init__(self, data, linking_length, np=None, verbose=False):
        self.data = data
        self.linking_length = linking_length

        # iterations
        self.iterations = 0
        perm = utils.empty(len(data), dtype='intp')
        head = utils.empty(len(data), dtype='intp')
        head[:] = numpy.arange(len(data), dtype='intp')
        op = 1 
        while op > 0:
            op = self._once(perm, head, np)
            self.iterations = self.iterations + 1
            if verbose:
                print 'FOF iteration', self.iterations, op
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

    def _once(self, perm, head, np):
        """ fof iteration,
            head[i] is the index of the head particle of the FOF group i
              is currently in
            perm is a scratch space for permutation;
               in each iteration head[i] is replaced with perm[head[i]]
        """ 
        tree = self.data.tree
        if np != 0:
            p = list(utils.divide_and_conquer(tree, tree, 10000))
        else:
            p = [(tree, tree)]

        #print 'p', len(p)
        ll = self.linking_length
        ops = [0]

        with utils.MapReduce(np=np) as pool:
            chunksize = 1024 * 1024
            # fill perm with no changes
            def init(i):
                s = slice(i, i + chunksize)
                a, b, c = s.indices(len(head))
                perm[s] = numpy.arange(a, b)
            pool.map(init, range(0, len(head), chunksize))

            # calculate perm, such that if two groups are 
            # merged, the head is set to the smaller particle index
            def work(iwork):
                n1, n2 = p[iwork]
                operations = [0]
                def process(r, i, j):
                    if len(r) == 0: return 
#                    print iwork, 'len(r)', len(r)

                    # update the head id; 
                    # only for those that would decrease
                    mask2 = head[i] > head[j]
                    i = i[mask2]
                    j = j[mask2]
                    ni = head[i]
                    nj = head[j]
                    # we will replace in head all ni-s to nj
                    # find the minimal replacement of ni
                    arg = numpy.lexsort((ni, -nj))
                    ni = ni[arg]
                    nj = nj[arg]
                    #  find the last item in each i
                    lasts = (ni[1:] != ni[:-1]).nonzero()[0]
                    ni = ni[lasts]
                    nj = nj[lasts]

                    #  write to each entry, once, in order
                    #  minimizing memory clashes from many ranks;
                    #  the algorithm is stable against racing
                    #  but it would slow down the cache if each rank
                    #  were directly writing.
                    with pool.critical:
                        mask = perm[ni] > nj
                        ni = ni[mask]
                        nj = nj[mask]
                        perm[ni] = nj
                        operations[0] += len(ni)
#                    print iwork, 'len(r)', len(i)
                n1.enum(n2, ll, process, bunch=10000 * 8)
#                print 'work', iwork, 'done'
                return operations[0]

            def reduce(op):
                #count number of operations
                #print ops[0]
                ops[0] += op

            pool.map(work, range(len(p)), reduce=reduce)

            # replace; this is done repeatedly
            # since it is possible we count a progenitor
            # into a merged progenitor.
            # in that case we do not want to do another
            # round of expensive tree walk
            def work2(i):
                s = slice(i, i + chunksize)
                N = 1
                while N > 0:
                    tmp = perm[head[s]]
                    N = (head[s] != tmp).sum()
                    head[s] = tmp
                     
            pool.map(work2, range(0, len(head), chunksize))
        return ops[0]

