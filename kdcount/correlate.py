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
    
    """
    def __init__(self, *args):
        """ the shape has one extra per edge
            0 is .le. min
            -1 is .g. max
            args are (min, max, Nbins)
            the first is for R
            centers is squezzed
        """
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

    def __call__(self, r, i, j, data1, data2):
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
    def __init__(self, Rmax, Nbins, Nmubins, observer):
        Binning.__init__(self, 
                (0, Rmax, Nbins),
                (-1, 1, Nmubins)
            )
        self.observer = numpy.array(observer)

    def __call__(self, r, i, j, data1, data2):
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        center = numpy.einsum('ij, ij->i', center, center) ** 0.5
        mu = dot / (center * r)
        mu[r == 0] = 10.0
        return self.linear(r, mu)

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

    def __init__(self, Rmax, Nbins, observer):
        self.Rmax = Rmax
        Binning.__init__(self,
            (0, Rmax, Nbins),
            (-Rmax, Rmax, 2 * Nbins)
            )
        self.observer = observer

    def __call__(self, r, i, j, data1, data2):
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
        return self.linear(sky, los)

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
    def __init__(self, Rmax, Nbins, logscale=False, Rmin=0):
        Binning.__init__(self, 
                (Rmin, Rmax, Nbins, logscale))
    def __call__(self, r, i, j, data1, data2):
        return self.linear(r)

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

        if the value has multiple components,
        return counts with be 'tuple', one item for each component
        """

        tree1 = data1.tree
        tree2 = data2.tree
        if np != 0:
            p = list(utils.divide_and_conquer(tree1, tree2, 10000))
        else:
            p = [(tree1, tree2)]
        linearshape = [-1] + list(binning.shape)

        if isinstance(data1, points) and isinstance(data2, points):
            fullshape = list(data1.subshape) + list(binning.shape)
        elif isinstance(data1, points) and isinstance(data2, field):
            sum2g = numpy.zeros(binning.shape, dtype='f8').reshape(linearshape)
            fullshape = list(data2.subshape) + list(binning.shape)
        elif isinstance(data1, field) and isinstance(data2, points):
            sum2g = numpy.zeros(binning.shape, dtype='f8').reshape(linearshape)
            fullshape = list(data1.subshape) + list(binning.shape)
        elif isinstance(data1, field) and isinstance(data2, field):
            assert data1.subshape == data2.subshape
            sum2g = numpy.zeros(binning.shape, dtype='f8').reshape(linearshape)
            fullshape = list(data1.subshape) + list(binning.shape)

        sum1g = numpy.zeros(fullshape, dtype='f8').reshape(linearshape)

        def work(i):
            n1, n2 = p[i]
            sum1 = numpy.zeros_like(sum1g)
            if isinstance(data1, points) and isinstance(data2, points):
                sum2 = 1.0
            else:
                sum2 = numpy.zeros_like(sum2g)
            def callback(r, i, j):
                dig = binning(r, i, j, data1, data2)
                if isinstance(data1, field) and isinstance(data2, field):
                    sum1ij = data1.wv(i) * data2.wv(j)
                    sum2ij = data1.w(i) * data2.w(j)
                    sum1ij = sum1ij.reshape(dig.size, -1)
                    for d in range(sum1.shape[0]):
                        sum1[d].flat [:] += utils.bincount(dig, 
                                sum1ij[:, d],
                                minlength=sum1[d].size)
                    sum2.flat [:] += utils.bincount(dig, 
                            sum2ij,
                            minlength=sum2.size)
                elif isinstance(data1, field) and isinstance(data2, points):
                    sum1ij = data1.wv(i) * data2.w(j)
                    sum2ij = data1.w(i) * data2.w(j)
                    for d in range(sum1.shape[0]):
                        sum1[d].flat [:] += utils.bincount(dig, 
                                sum1ij[:, d],
                                minlength=sum1[d].size)
                    sum2.flat [:] += utils.bincount(dig, 
                                sum2ij,
                                minlength=sum2.size)
                elif isinstance(data1, points) and isinstance(data2, field):
                    sum1ij = data1.w(i) * data2.wv(j)
                    sum2ij = data1.w(i) * data2.w(j)
                    for d in range(sum1.shape[0]):
                        sum1[d].flat [:] += utils.bincount(dig, 
                                sum1ij[:, d],
                                minlength=sum1[d].size)
                    sum2.flat [:] += utils.bincount(dig, 
                                sum2ij,
                                minlength=sum2.size)
                elif isinstance(data1, points) and isinstance(data2, points):
                    sum1ij = data1.w(i) * data2.w(j)
                    sum1.flat [:] += utils.bincount(dig, 
                            sum1ij,
                            minlength=sum1.size)
            if usefast and type(binning) is RBinning \
                and isinstance(data1, points) \
                and isinstance(data2, points) \
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
            if not (isinstance(data1, points) and isinstance(data2, points)):
                sum2g[...] += sum2
        with utils.MapReduce(np=np) as pool:
            pool.map(work, range(len(p)), reduce=reduce)

        self.fullsum1 = sum1g.reshape(fullshape).copy()
        self.sum1 = self.fullsum1[[Ellipsis] + [slice(1, -1)] * binning.Ndim]

        if not (isinstance(data1, points) and isinstance(data2, points)):
            self.fullsum2 = sum2g.reshape(binning.shape).copy()
            self.sum2 = self.fullsum2[ [slice(1, -1)] * binning.Ndim]

        self.binning = binning
        self.edges = binning.edges
        self.centers = binning.centers

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
