import kdcount
import numpy
try:
    from sharedmem import MapReduce
except ImportError:
    class MapReduce(object):
        def __init__(self, np=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, work, items, reduce=None):
            if reduce is not None: 
                callreduce = lambda r: \
                    reduce(*r) if isinstance(r, tuple) \
                        else reduce(r)
            else:
                callreduce = lambda r: r
            return [callreduce(work(i)) for i in items]

class dataset(object):
    def __init__(self, pos, boxsize, extra):
        """ create a dataset object for points located at pos in a boxsize.
            points is of (Npoints, Ndim)
            boxsize will be broadcasted to the dimension of points. 
            extra can be accessed as dataset.extra.
        """
        self.pos = pos
        self.tree = kdcount.build(self.pos, boxsize=boxsize)
        self.extra = extra

    def __len__(self):
        return len(self.pos)

class points(dataset):
    def __init__(self, pos, weight=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, boxsize, extra)
        self._weight = weight
        if weight is not None:
            assert len(weight.shape) == 1
            self.norm = weight.sum(axis=0)
        else:
            self.norm = len(pos) * 1.0
        self.subshape = ()

    def w(self, index):
        if self._weight is None:
            return 1.0
        else:
            return self._weight[index]

class field(dataset):
    def __init__(self, pos, value, weight=None, boxsize=None, extra={}):
        dataset.__init__(self, pos, boxsize, extra)
        self._weight = weight
        if weight is not None:
            self._value = value * weight
        else:
            self._value = value
        self.subshape = value.shape[1:]
    def wv(self, index):
        return self._value[index]
    def w(self, index):
        if self._weight is None:
            return 1.0
        else:
            return self._weight[index]

class Binning(object):
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

    def __call__(self, r, i, j):
        raise UnimplementedError()

class RmuBinning(Binning):
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
    """ the bins will be 
        [sky, los]
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
    def __init__(self, Rmax, Nbins, logscale=False, Rmin=0):
        Binning.__init__(self, 
                (Rmin, Rmax, Nbins, logscale))
    def __call__(self, r, i, j, data1, data2):
        return self.linear(r)
def mybincount(dig, weight, minlength):
    if numpy.isscalar(weight):
        return numpy.bincount(dig, minlength=minlength) * weight
    else:
        return numpy.bincount(dig, weight, minlength)

class paircount(object):
    """ 
        a paircount object has the following attributes:
        sum1 :    the numerator in the correlator
        sum2 :    the denominator in the correlator
        corr :    sum1 / sum2

        for points x points: 
               sum1 = sum( w1 w2 )
               sum2 = 1.0 
        for field x points:
               sum1 = sum( w1 w2 v1)
               sum2 = sum( w1 w2)
        for field x field:
               sum1 = sum( w1 w2 v1 v2)
               sum2 = sum( w1 w2)

        with this convention the usual form of landy-sarley
        (DD.sum1 -2r DR.sum1 + r2 RR.sum1) / (r2 RR.sum1) 
        (with r = sum(wD) / sum(wR))

        centers : the centers of the corresponding corr bin
                  centers = (X, Y, ....)
                  len(X) == corr.shape[0], len(Y) == corr.shape[1]
        binning : the binning object to create this paircount 
        edges :   the edges of the corr bins.

        fullcorr : the full correlation function with outliners 
                    len(X) == corr.shape[0] + 2 
        fullsum1 : full version of sum1
        fullsum2 : full version of sum2
    """
    def __init__(self, data1, data2, binning, np=None):
        """
        binning is an instance of Binning, (eg, RBinning, RmuBinning)

        if the value has multiple components,
        return counts with be 'tuple', one item for each component
        """

        tree1 = data1.tree
        tree2 = data2.tree
        if np != 0:
            p = list(kdcount.divide_and_conquer(tree1, tree2, 10000))
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
            for r, i, j in n1.enumiter(n2, binning.Rmax):
                dig = binning(r, i, j, data1, data2)
                if isinstance(data1, field) and isinstance(data2, field):
                    sum1ij = data1.wv(i) * data2.wv(j)
                    sum2ij = data1.w(i) * data2.w(j)
                    sum1ij = sum1ij.reshape(dig.size, -1)
                    for d in range(sum1.shape[0]):
                        sum1[d].flat [:] += mybincount(dig, 
                                sum1ij[:, d],
                                minlength=sum1[d].size)
                    sum2.flat [:] += mybincount(dig, 
                            sum2ij,
                            minlength=sum2.size)
                elif isinstance(data1, field) and isinstance(data2, points):
                    sum1ij = data1.wv(i) * data2.w(j)
                    sum2ij = data1.w(i) * data2.w(j)
                    print sum1ij, sum2ij
                    for d in range(sum1.shape[0]):
                        sum1[d].flat [:] += mybincount(dig, 
                                sum1ij[:, d],
                                minlength=sum1[d].size)
                    sum2.flat [:] += mybincount(dig, 
                                sum2ij,
                                minlength=sum2.size)
                elif isinstance(data1, points) and isinstance(data2, field):
                    sum1ij = data1.w(i) * data2.wv(j)
                    sum2ij = data1.w(i) * data2.w(j)
                    for d in range(sum1.shape[0]):
                        sum1[d].flat [:] += mybincount(dig, 
                                sum1ij[:, d],
                                minlength=sum1[d].size)
                    sum2.flat [:] += mybincount(dig, 
                                sum2ij,
                                minlength=sum2.size)
                elif isinstance(data1, points) and isinstance(data2, points):
                    sum1ij = data1.w(i) * data2.w(j)
                    sum1.flat [:] += mybincount(dig, 
                            sum1ij,
                            minlength=sum1.size)
            return sum1, sum2
        def reduce(sum1, sum2):
            sum1g[...] += sum1
            if not (isinstance(data1, points) and isinstance(data2, points)):
                sum2g[...] += sum2
        with MapReduce(np=np) as pool:
            pool.map(work, range(len(p)), reduce=reduce)

        self.fullsum1 = sum1g.reshape(fullshape).copy()
        self.sum1 = self.fullsum1[[Ellipsis] + [slice(1, -1)] * binning.Ndim]

        if not (isinstance(data1, points) and isinstance(data2, points)):
            self.fullsum2 = sum2g.reshape(binning.shape).copy()
            self.sum2 = self.fullsum2[ [slice(1, -1)] * binning.Ndim]

        self.binning = binning
        self.edges = binning.edges
        self.centers = binning.centers

def main():
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

def main2():
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    print 'read'
    print sim
    print sim.mean(dtype='f8'), sim.max()
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.1
    value = numpy.tile(sim[sample], (2, 1)).T
#    value = sim[sample]
    data = field(pos[sample], value=value)
    print 'data ready'
    binning = RBinning(0.1, 40)
    DD = paircount(data, data, binning)

    return DD.centers, DD.sum1 / DD.sum2

def main3():
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    print 'read'
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.2
    value = numpy.tile(sim[sample], (2, 1)).T
    data = field(pos[sample], value=value)
    print 'data ready'
    DD = paircount(data, data, RmuBinning(0.10, 8, 20, 0.5))
    return DD
