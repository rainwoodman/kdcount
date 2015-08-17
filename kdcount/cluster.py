""" 
Clustering with KDTree

This module implements Friend-of-Friend clustering with a KDtree as
:py:class:`fof`.

The Friend-of-Friend clustering algorithm is implemented (Davis et al. 1985 ADSlink:)
The algorithm is commonly used in astronomy and cosmology: all points
that are within a given 'linking_length' are clustered into one object.

"""

import numpy
from .models import dataset
from . import utils
from sys import stdout

class fof(object):
    """ 
    Friend of Friend clustering

    Attributes
    ----------
    data    : :py:class:`kdcount.models.dataset`
        data set (positions of particles complied into a KD-Tree
    linking_length : float
        linking length, in data units
    np     : int
        parallel processes to use (0 to disable)
    verbose : boolean
        print some verbose information

    N   : int
        number of clusters identified
    labels : array_like
        the label (cluster id) of each object 
    length : array_like
        number of particles per cluster 
    offset : array_like
        offset of the first particle in indices
    indices : array_like
        index of particles indices[offset[i]:length[i]] is the indices
        of particles in cluster i.

    """    
    def __init__(self, data, linking_length, np=None, verbose=False):
        self.data = data
        self.linking_length = linking_length

        # iterations
        self.iterations = 0
        perm = utils.empty(len(data), dtype='intp')
        head = utils.empty(len(data), dtype='intp')
        head[:] = numpy.arange(len(data), dtype='intp')


        llfactor = 8
        while llfactor > 0:
            op = 1 
            ll = self.linking_length / llfactor
            while op > 0:
                op = self._once(perm, head, np, ll)
                self.iterations = self.iterations + 1
                if verbose:
                    print('FOF iteration', self.iterations, op, llfactor)
                    stdout.flush()
                if llfactor != 1:
                    break
            llfactor = llfactor // 2

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

    def _once(self, perm, head, np, ll):
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
                def process(r, i, j, head=head, perm=perm, pool=pool,
                ):
                    if len(r) == 0: return 
#                    print iwork, 'len(r)', len(r)

                    ni = head[i]
                    nj = head[j]

                    mask = (r <= ll) & (ni > nj)

                    if not mask.any(): return

                    # we will replace in head all ni-s to nj
                    ni = ni[mask]
                    nj = nj[mask]
                        
                    # find the minimal replacement of ni
                    arg = numpy.lexsort((ni, -nj))

                    ni = ni[arg]
                    nj = nj[arg]
                    #  find the last item in each i
                    mask3 = numpy.empty(len(ni), '?')
                    mask3[:-1] = ni[1:] != ni[:-1]
                    mask3[-1] = True

                    ni = ni[mask3]
                    nj = nj[mask3]

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

#                    print iwork, 'len(r)', len(i)
                n1.enum(n2, ll, process, bunch=1024 * 80)
#                print 'work', iwork, 'done'
                return

            pool.map(work, range(len(p)))

            # replace; this is done repeatedly
            # since it is possible we count a progenitor
            # into a merged progenitor.
            # in that case we do not want to do another
            # round of expensive tree walk
            def work2(i):
                s = slice(i, i + chunksize)
                ops = 0
                while True:
                    tmp = perm[head[s]]
                    changed = head[s] != tmp
                    if changed.any():
                        ops += int(changed.sum())
                        head[s] = tmp
                    else:
                        break
                return ops         
            ops = pool.map(work2, range(0, len(head), chunksize))
        return sum(ops)

