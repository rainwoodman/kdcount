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
    def __init__(self, data, linking_length, np=None):
        self.data = data
        self.linking_length = linking_length

        # iterations
        self.iterations = 0
        head = utils.empty(len(data), dtype='intp')

        # this will set the head to a particle index 
        # need to convert this to a feature id with unique.
        data.tree.root.fof(linking_length, head)

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
            weights = self.data.weights
        return utils.bincount(self.labels, weights, self.N)

    def center(self, weights=None):
        """ return the center of each object """
        if weights is None:
            weights = self.data.weights
        mass = utils.bincount(self.labels, weights, self.N)
        cp = numpy.empty((len(mass), self.data.pos.shape[-1]), 'f8')
        for d in range(self.data.pos.shape[-1]):
            cp[..., d] = utils.bincount(self.labels, weights *
                    self.data.pos[..., d], self.N)
            cp[..., d] /= mass
        return cp


