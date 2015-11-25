"""
Correlation function (pair counting) with KDTree.

Pair counting is the basic algorithm to calculate correlation functions.
Correlation function is a commonly used metric in cosmology to measure
the clustering of matter, or the growth of large scale structure in the universe.

We implement :py:class:`paircount` for pair counting. Since this is a discrete
estimator, the binning is modeled by subclasses of :py:class:`Binning`. For example

- :py:class:`RBinning` 
- :py:class:`RmuBinning`
- :py:class:`XYBinning`
- :py:class: `FlatSkyBinning`
- :py:class: `FlatSkyMultipoleBinning`

kdcount takes two types of input data: 'point' and 'field'. 

:py:class:`kdcount.models.points` describes data with position and weight. For example, galaxies and
quasars are point data. 
point.pos is a row array of the positions of the points; other fields are
used internally.
point.extra is the extra properties that can be used in the Binning. One use
is to exclude the Lyman-alpha pixels and Quasars from the same sightline. 

:py:class:`kdcount.models.field` describes a continious field sampled at given positions, each sample
with a weight; a notorious example is the over-flux field in Lyman-alpha forest
it is a proxy of the over-density field sampled along quasar sightlines. 

In the Python Interface, to count, one has to define the 'binning' scheme, by
subclassing :py:class:`Binning`. Binning describes a multi-dimension binning
scheme. The dimensions can be derived, for example, the norm of the spatial
separation can be a dimension the same way as the 'x' separation. For example, 
see :py:class:`RmuBinning`.


"""
import numpy

# local imports
from .models import points, field
from . import utils 


def compute_sum_values(i, j, data1, data2):
    """
    Return the sum1_ij and sum2_ij values given
    the input indices and data instances.
    
    Notes
    -----
    This is called in `Binning.__call__` to compute
    the `sum1` and `sum2` contributions for indices `(i,j)`
    
    Parameters
    ----------
    i,j : array_like
        the bin indices for these pairs
    data1, data2 : `points`, `field` instances
        the two `points` or `field` objects
        
    Returns
    -------
    sum1_ij, sum2_ij : float, array_like (N,...)
        contributions to sum1, sum2 -- either a float or array 
        of shape (N, ...) where N is the length of `i`, `j`
    """
    sum1_ij = 1.
    for idx, d in zip([i,j], [data1, data2]):
        if isinstance(d, field): sum1_ij *= d.wv(idx)
        elif isinstance(d, points): sum1_ij *= d.w(idx)
        else:
            raise NotImplementedError("data type not recognized")
    sum2_ij = data1.w(i) * data2.w(j)

    return sum1_ij, sum2_ij

class Binning(object):
    """
    Binning of the correlation function. Pairs whose distance is with-in a bin
    is counted towards the bin.
    
    Attributes
    ----------
    dims    :  array_like
        internal; descriptors of binning dimensions.
    edges   :  array_like
        edges of bins per dimension
    centers :  array_like
        centers of bins per dimension; currently it is the 
        mid point of the edges.
    compute_mean_coords : bool, optional
        If `True`, store and compute the mean coordinate values
        in the __call__ function. Default is `False`
    
    """
    def __init__(self, *args, **kwargs):
        """ the shape has one extra per edge
            0 is .le. min
            -1 is .g. max
            args are (min, max, Nbins)
            the first is for R
            centers is squezzed
        """
        self.compute_mean_coords = kwargs.get('compute_mean_coords', False)
        
        self.dims = numpy.empty(len(args),
                dtype=[
                    ('inv', 'f8'),
                    ('min', 'f8'),
                    ('max', 'f8'),
                    ('N', 'i4'),
                    ('logscale', '?')
                    ])
        self.min = self.dims['min']
        self.max = self.dims['max']
        self.N = self.dims['N']
        self.inv = self.dims['inv']
        self.logscale = self.dims['logscale']
        self.edges = []
        self.centers = []
        
        for i, dim in enumerate(args):
            if len(dim) == 3:
                min, max, Nbins = dim
                log = False
            else:
                min, max, Nbins, log = dim
            self.N[i] = Nbins
            self.min[i] = min
            self.max[i] = max
            self.logscale[i] = log
            if log:
                self.inv[i] = Nbins * 1.0 / numpy.log10(max / min)
            else:
                self.inv[i] = Nbins * 1.0 / (max - min)
            edge = numpy.arange(Nbins + 1) * 1.0 / self.inv[i]
            if log:
                edge = 10 ** edge * min
                center = (edge[1:] * edge[:-1]) ** 0.5
            else:
                edge = edge + min
                center = (edge[1:] + edge[:-1]) * 0.5
            self.edges.append(edge)
            self.centers.append(center)

        self.Rmax = self.max[0]
        self.Ndim = len(args)
        self.shape = self.N + 2
        if self.Ndim == 1:
            self.edges = self.edges[0]
            self.centers = self.centers[0]
            
        # for storing the mean values in each bin
        # computed when pair counting
        if self.compute_mean_coords:
            self.mean_centers_sum = []
            for idim in range(self.Ndim):
                self.mean_centers_sum.append(numpy.zeros(self.shape))        
            self.pair_counts = numpy.zeros(self.shape)
                   
    def linear(self, *args):
        """ 
        Linearize bin indices.
        
        This function is called by subclasses. Refer to the source
        code of :py:class:`RBinning` for an example.

        Parameters
        ----------
        args    : list
            a list of bin index, (xi, yi, zi, ..) 
        
        Returns
        -------
        linearlized bin index
        """
        integer = numpy.empty(len(args[0]), ('i8', (self.Ndim,))).T
        for d in range(self.Ndim):
            if self.logscale[d]:
                x = args[d].copy()
                x[x == 0] = self.min[d] * 0.9
                x = numpy.log10(x / self.min[d])
            else:
                x = args[d] - self.min[d]
            integer[d] = numpy.ceil(x * self.inv[d])
        return numpy.ravel_multi_index(integer, self.shape, mode='clip')

    def digitize(self, r, i, j, data1, data2):
        """
        Calculate the bin number of pairs separated by distances r, 
        Use :py:meth:`linear` to convert from multi-dimension bin index to
        linear index.
 
        Parameters
        ----------
        r   : array_like
            separation

        i, j : array_like
            index (i, j) of pairs. 
        data1, data2 :
            The position of first point is data1.pos[i], the position of second point is
            data2.pos[j]. 

        """
        raise UnimplementedError()
    
    def __call__(self, r, i, j, data1, data2, sum1, sum2):
        """
        The main function that digitizes the pair counts, 
        calls bincount for the appropriate `sum1` and `sum2`
        values, and adds them to the input arrays
        """
        # the summation values for this (r,i,j)
        sum1_ij, sum2_ij = compute_sum_values(i, j, data1, data2)
        
        # digitize
        dig = self.digitize(r, i, j, data1, data2)
        
        # sum 1
        if not numpy.isscalar(sum1_ij) and sum1_ij.ndim > 1:
            for d in range(sum1.shape[0]):
                sum1[d].flat[:] += utils.bincount(dig, sum1_ij[...,d], minlength=sum1[d].size)
        else:
            sum1.flat[:] += utils.bincount(dig, sum1_ij, minlength=sum1.size)
            
        # sum 2, if both data are not points
        if not numpy.isscalar(sum2):
            sum2.flat[:] += utils.bincount(dig, sum2_ij, minlength=sum2.size)
                    
    def sum_shapes(self, data1, data2):
        """
        Return the shapes of the summation arrays, 
        given the input data and shape of the bins
        """
        # the linear shape (put extra dimensions first)
        linearshape = [-1] + list(self.shape)
    
        # determine the full shape
        subshapes = [list(d.subshape) for d in [data1, data2] if isinstance(d, field)]
        subshape = []
        if len(subshapes) == 2:
            assert subshapes[0] == subshapes[1]
            subshape = subshapes[0]
        elif len(subshapes) == 1:
            subshape = subshapes[0]
        fullshape = subshape + list(self.shape)
        
        return linearshape, fullshape
        
    def update_mean_coords(self, dig, *args):
        """
        Update the mean coordinate sums
        """
        if not self.compute_mean_coords:
            return
        
        self.pair_counts.flat[:] += utils.bincount(dig, 1., minlength=self.pair_counts.size)
        for idim in range(self.Ndim):
            size = self.mean_centers_sum[idim].size
            self.mean_centers_sum[idim].flat[:] += utils.bincount(dig, args[idim], minlength=size)
            
        
class RmuBinning(Binning):
    """
    Binning in R and mu (angular along line of sight)
    mu = cos(theta), relative to line of sight from a given observer. 

    Parameters
    ----------
    Rmax     : float
        max radius to go to
    Nbins    : int
        number of bins in R direction.
    Nmubins    : int
        number of bins in mu direction.
    observer   : array_like (Ndim)
        location of the observer (for line of sight) 

    """
    def __init__(self, Rmax, Nbins, Nmubins, observer, **kwargs):
        Binning.__init__(self, 
                (0, Rmax, Nbins),
                (-1, 1, Nmubins),
                **kwargs
            )
        self.observer = numpy.array(observer)

    def digitize(self, r, i, j, data1, data2):
        
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        center = numpy.einsum('ij, ij->i', center, center) ** 0.5
        mu = dot / (center * r)
        mu[r == 0] = 10.0
        dig = self.linear(r, mu)
        
        # update the mean coords
        self.update_mean_coords(dig, r, mu)
        
        return dig

class XYBinning(Binning):
    """ 
    Binning along Sky-Lineofsight directions.

    The bins are be (sky, los)

    Parameters
    ----------
    Rmax     : float
        max radius to go to
    Nbins    : int
        number of bins in each direction.
    observer   : array_like (Ndim)
        location of the observer (for line of sight) 

    Notes
    -----
    with numpy imshow , the second axis los, will be vertical
            with imshow( ..T,) the sky will be vertical.
    """

    def __init__(self, Rmax, Nbins, observer, **kwargs):
        self.Rmax = Rmax
        Binning.__init__(self,
            (0, Rmax, Nbins),
            (-Rmax, Rmax, 2 * Nbins), 
            **kwargs
            )
        self.observer = observer

    def digitize(self, r, i, j, data1, data2):
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        center2 = numpy.einsum('ij, ij->i', center, center)
        los = dot / center2 ** 0.5
        dr2 = numpy.einsum('ij, ij->i', dr, dr)
        x2 = numpy.abs(dr2 - los ** 2)
        sky = x2 ** 0.5
        
        dig = self.linear(sky, los)
        
        # update the mean coords
        self.update_mean_coords(dig, sky, los)
        
        return dig

class RBinning(Binning):
    """ 
    Binning along radial direction.

    Parameters
    ----------
    Rmax     : float
        max radius to go to
    Nbins    : int
        number of bins in each direction.

    """
    def __init__(self, Rmax, Nbins, logscale=False, Rmin=0, **kwargs):
        rbins = (Rmin, Rmax, Nbins, logscale)
        Binning.__init__(self, rbins, **kwargs)
        
    def digitize(self, r, i, j, data1, data2):
        
        # linear bins
        dig = self.linear(r)
        
        # update the mean coords
        self.update_mean_coords(dig, r)

        return dig
        

class FlatSkyMultipoleBinning(Binning):
    """
    Binning in R and `ell`, the multipole number, in the 
    flat sky approximation, such that all pairs have the 
    same line-of-sight, which is taken to be the axis specified 
    by the `los` parameter (default is the last dimension)
    

    Parameters
    ----------
    rmax : float
        the maximum radius to measure to
    Nr : int
        the number of bins in `r` direction.
    ells : list of int
        the multipole numbers to compute
    los : int, {0, 1, 2}
        the axis to treat as the line-of-sight
    """
    def __init__(self, rmax, Nr, ells, los, **kwargs):
        from scipy.special import legendre 
        
        rbins = (0, rmax, Nr)
        Binning.__init__(self, rbins, **kwargs)
        
        self.los = los
        self.ells = numpy.array(ells)
        self.legendre = [legendre(l) for l in self.ells]
                
    
    def digitize(self, r, i, j, data1, data2):
        
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        
        # parallel separation
        d_par = (r1-r2)[:,self.los]
        
        # enforce periodic boundary conditions
        L = data1.boxsize[self.los]
        d_par[d_par > L*0.5] -= L
        d_par[d_par <= -L*0.5] += L
        
        # mu
        with numpy.errstate(invalid='ignore'):
            mu = d_par / r
    
        # linear bin index and weights
        dig = self.linear(r)
        w = numpy.array([leg(mu) for leg in self.legendre]) # shape should be (N_ell, len(r1))
        w *= (2*self.ells+1)[:,None]
        
        # update the mean coords
        self.update_mean_coords(dig, r)
        
        return dig, w
        
    def __call__(self, r, i, j, data1, data2, sum1, sum2):
        """
        Overloaded function to compute the sums as a function
        of `ell` in addition to `r`
        
        Notes
        -----
        valid only when `data1` and `data2` are both `points` instances
        """        
        pts_only = isinstance(data1, points) and isinstance(data2, points)
        if not pts_only:
            name = self.__class__.__name__
            raise NotImplementedError("%s only valid when correlating two `points` instances" %name)
            
        # the summation values for this (r,i,j)
        sum1_ij, sum2_ij = compute_sum_values(i, j, data1, data2)
        
        # digitize
        dig, weights = self.digitize(r, i, j, data1, data2)
                
        # sum 1
        for iell in range(len(self.ells)):
            tosum = (weights*sum1_ij)[iell,...]
            sum1[iell, ...].flat[:] += utils.bincount(dig, tosum, minlength=sum1[iell,...].size)
                    
    def sum_shapes(self, data1, data2):
        """
        Prepend the shape of `ells` to the summation arrays
        """
        linearshape, fullshape = Binning.sum_shapes(self, data1, data2)
        fullshape = [len(self.ells)] + fullshape
        
        return linearshape, fullshape

class FlatSkyBinning(Binning):
    """
    Binning in R and mu, in the flat sky approximation, such 
    that all pairs have the same line-of-sight, which is 
    taken to be the axis specified by the `los` parameter 
    (default is the last dimension)
    

    Parameters
    ----------
    rmax : float
        the maximum radius to measure to
    Nr : int
        the number of bins in `r` direction.
    Nmu : int
        the number of bins in `mu` direction.
    los : int, {0, 1, 2}
        the axis to treat as the line-of-sight
    """
    def __init__(self, rmax, Nr, Nmu, los, **kwargs):
        rbins = (0, rmax, Nr)
        mubins = (-1, 1, Nmu)
        Binning.__init__(self, rbins, mubins, **kwargs)
        self.los = los

    def digitize(self, r, i, j, data1, data2):
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        
        # parallel separation
        d_par = (r1-r2)[:,self.los]
        
        # enforce periodic boundary conditions
        L = data1.boxsize[self.los]
        d_par[d_par > L*0.5] -= L
        d_par[d_par <= -L*0.5] += L
        
        # mu
        with numpy.errstate(invalid='ignore'):
            mu = d_par / r
        mu[r == 0] = 10.0 # ignore self pairs by setting mu out of bounds
        
        # linear bin index
        dig = self.linear(r, mu)
        
        # update the mean coords
        self.update_mean_coords(dig, r, mu)
        
        return dig

class paircount(object):
    """ 
    Paircounting via a KD-tree, on two data sets.

    Attributes
    ----------
    sum1 :  array_like
        the numerator in the correlator
    sum2 :  array_like
        the denominator in the correlator
    centers : list
        the centers of the corresponding corr bin, one item per 
        binning direction.
    edges :   list
        the edges of the corresponding corr bin, one item per 
        binning direction.
    binning : :py:class:`Binning`
        binning object of this paircount 
    data1   : :py:class:`dataset`
        input data set1. It can be either 
        :py:class:`field` for discrete sampling of a continuous
        field, or :py:class:`kdcount.models.points` for a point set.
    data2   : :py:class:`dataset`
        input data set2, see above.
    np : int
        number of parallel processes. set to 0 to disable parallelism

    Notes
    -----
    The value of sum1 and sum2 depends on the types of input 

    For :py:class:`kdcount.models.points` and :py:class:`kdcount.models.points`: 
      - sum1 is the per bin sum of products of weights 
      - sum2 is always 1.0 

    For :py:class:`kdcount.models.field` and :py:class:`kdcount.models.points`:
      - sum1 is the per bin sum of products of weights and the field value
      - sum2 is the per bin sum of products of weights

    For :py:class:`kdcount.models.field` and :py:class:`kdcount.models.field`:
      - sum1 is the per bin sum of products of weights and the field value
        (one value per field)
      - sum2 is the per bin sum of products of weights

    With this convention the usual form of Landy-Salay estimator is (
    for points x points:

        (DD.sum1 -2r DR.sum1 + r2 RR.sum1) / (r2 RR.sum1) 

        with r = sum(wD) / sum(wR)

    """
    def __init__(self, data1, data2, binning, usefast=True, np=None):
        """
        binning is an instance of Binning, (eg, RBinning, RmuBinning)

        Notes
        -----
        *   if the value has multiple components, return counts with be 'tuple', 
            one item for each component
        *   if `binning.compute_mean_coords` is `True`, then `meancenters` will hold
            the mean coordinate value in each bin. Cannot have `usefast = True`
            and `binning.compute_mean_coords = True`
        """
        if usefast and binning.compute_mean_coords:
            raise NotImplementedError("cannot currently compute bin centers and use the `fast` algorithm")
        
        tree1 = data1.tree
        tree2 = data2.tree
        if np != 0:
            p = list(utils.divide_and_conquer(tree1, tree2, 10000))
        else:
            p = [(tree1, tree2)]
        pts_only = isinstance(data1, points) and isinstance(data2, points)

        # initialize arrays to hold total sum1 and sum2
        # grabbing the desired shapes from the binning instance
        linearshape, fullshape = binning.sum_shapes(data1, data2)
        sum1g = numpy.zeros(fullshape, dtype='f8').reshape(linearshape)
        if not pts_only:
            sum2g = numpy.zeros(binning.shape, dtype='f8').reshape(linearshape)
        
        # initialize arrays for computing mean coords
        N = None; centers = None
        if binning.compute_mean_coords:
            N = numpy.zeros_like(binning.pair_counts)
            centers = [numpy.zeros(binning.shape) for i in range(binning.Ndim)]
        
        def work(i):
            
            n1, n2 = p[i]
            
            # initialize the total arrays for this process
            sum1 = numpy.zeros_like(sum1g)
            sum2 = 1.
            if not pts_only:
                sum2 = numpy.zeros_like(sum2g)
            
            def callback(r, i, j):
                
                # just call the binning function, passing the 
                # sum arrays to fill in
                binning(r, i, j, data1, data2, sum1, sum2)
                                
            if usefast and type(binning) is RBinning and pts_only \
                and data1._weights is None \
                and data2._weights is None :
                counts, weights = data1.tree.count(data2.tree, binning.edges)
                d = numpy.diff(counts)
                sum1[0, 0] = counts[0]
                sum1[0, 1:-1] += d
            else:
                n1.enum(n2, binning.Rmax, process=callback)

            return sum1, sum2
       
        def reduce(sum1, sum2):
            sum1g[...] += sum1
            if not pts_only:
                sum2g[...] += sum2
            
            if binning.compute_mean_coords:
                N[...] += binning.pair_counts
                for i in range(binning.Ndim):
                    centers[i][...] += binning.mean_centers_sum[i]
                
        with utils.MapReduce(np=np) as pool:
            pool.map(work, range(len(p)), reduce=reduce)

        self.fullsum1 = sum1g.reshape(fullshape).copy()
        self.sum1 = self.fullsum1[[Ellipsis] + [slice(1, -1)] * binning.Ndim]

        if not pts_only:
            self.fullsum2 = sum2g.reshape(binning.shape).copy()
            self.sum2 = self.fullsum2[ [slice(1, -1)] * binning.Ndim]

        # finalize by attaching necessary binning info
        self.__finalize__(binning, centers, N)
        
    def __finalize__(self, binning, centers_sum, N):
        """
        Finalize the pair counting by attaching attributes of 
        `binning` to the `paircount` instance
        """
        self.binning = binning
        self.edges = binning.edges
        self.centers = binning.centers
        
        # add the mean centers info
        if binning.compute_mean_coords:
            
            # store the full sum too
            sl = [slice(1, -1)] * binning.Ndim
            self.pair_counts = N[sl]
            self.mean_centers_sum = []
        
            # do the division too
            self.mean_centers = []
            with numpy.errstate(invalid='ignore'):
                for i in range(binning.Ndim):
                    self.mean_centers_sum.append(centers_sum[i][sl])
                    y = self.mean_centers_sum[-1] / self.pair_counts
                    self.mean_centers.append(y)
                    
            if binning.Ndim == 1:
                self.mean_centers = self.mean_centers[0]

#------------------------------------------------------------------------------
# main functions for testing
#------------------------------------------------------------------------------
def _main():
    pm = numpy.fromfile('A00_hodfit.raw').reshape(-1, 8)[::1, :3]
    wm = numpy.ones(len(pm))
    martin = points(pm, wm)
    pr = numpy.random.uniform(size=(1000000, 3))
    wr = numpy.ones(len(pr))
    random = points(pr, wr)
    binning = RBinning(0.1, 40)
    DR = paircount(martin, random, binning)
    DD = paircount(martin, martin, binning)
    RR = paircount(random, random, binning)
    r = martin.norm / random.norm
    return binning.centers, (DD.sum1 - 
            2 * r * DR.sum1 + r ** 2 * RR.sum1) / (r ** 2 * RR.sum1)

def _main2():
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.1
    value = numpy.tile(sim[sample], (2, 1)).T
#    value = sim[sample]
    data = field(pos[sample], value=value)
    print('data ready')
    binning = RBinning(0.1, 40)
    DD = paircount(data, data, binning)

    return DD.centers, DD.sum1 / DD.sum2

def _main3():
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.2
    value = numpy.tile(sim[sample], (2, 1)).T
    data = field(pos[sample], value=value)
    print('data ready')
    DD = paircount(data, data, RmuBinning(0.10, 8, 20, 0.5))
    return DD
