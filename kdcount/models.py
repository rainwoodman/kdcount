import numpy
import kdcount
from .utils import constant_array

__all__ = ['dataset', 'points', 'field']

class dataset(object):
    """
    A data set with a KD-tree 
    
    The class is directly used with :py:class:`kdcount.cluster.fof` in 
    friend-of-friend clustering.

    Useful subclasses for :py:class:`kdcount.correlate.paircount` are
    :py:class:`points`, and :py:class:`field`

    Attributes
    ----------
    pos : array_like (Npoints, Ndim)
        position of sample points
    weights : array_like
        weight of objects, default is 1.0. Not to be confused with :code:`values`.
    boxsize : float
        if not None, a periodic boundary is assumed, and boxsize is the size of periodic box. 
    extra : dict
        extra properties.
 
    """
    def __init__(self, pos, weights=None, boxsize=None, extra={}):
        """ create a dataset object for points located at pos in a boxsize.
            points is of (Npoints, Ndim)
            boxsize will be broadcasted to the dimension of points. 
            extra can be accessed as dataset.extra.
        """
        self.pos = pos
        self.boxsize = boxsize
        self.tree = kdcount.KDTree(self.pos, boxsize=boxsize)
        self.extra = extra

        if weights is None:
            weights = constant_array(len(self.pos))
            weights.value[...] = 1

        self.weights = weights
        self.attr = kdcount.KDAttr(self.tree, weights)

    def __len__(self):
        return len(self.pos)

class points(dataset):
    """ 
    Point-wise data set
       
    Examples are galaxies, halos. These objects come with a position and a
    weight, and are discrete representation of the underlying density field.

    """
    def __init__(self, pos, weights=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, weights, boxsize, extra)

        self.norm = self.weights.sum(axis=0)
        self.subshape = ()

class field(dataset):
    """ 
    Discrte sampling data set
       
    Examples are Lyman-alpha forest transmission fractions, over-density fields. 
    These objects come with a position, a value, and a weight. The value
    is the underlying field integrated over some sampling kernel.

    Attributes
    ----------
    pos     : array_like
        sample points
    wvalue   : array_like
        the weighted value at pos.

    """
    def __init__(self, pos, value, weights=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, weights, boxsize, extra)
        self.wvalue = self.weights * value
        self.subshape = value.shape[1:]
        self.wvattr = kdcount.KDAttr(self.tree, self.wvalue)
