import kdcount
import numpy
try:
    from sharedmem import Pool
except ImportError:
    class Pool(object):
        def __init__(self, use_threads):
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
    def __init__(self, pos, boxsize):
        self.pos = pos
        self.tree = kdcount.build(self.pos, boxsize=boxsize)

class points(dataset):
    def __init__(self, pos, weight=None, boxsize=None):
        dataset.__init__(self, pos, boxsize)
        self._weight = weight
        if weight is not None:
            self.norm = weight.sum(axis=0)
            self.subshape = weight.shape[1:]
        else:
            self.norm = len(pos) * 1.0
            self.subshape = ()

    def num(self, index):
        if self._weight is None:
            return numpy.ones(len(index))
        else:
            return self._weight[index]
    def denom(self, index):
        return 1.0

class field(dataset):
    def __init__(self, pos, value, weight=None, boxsize=None):
        dataset.__init__(self, pos, boxsize)
        self._weight = weight
        if weight is not None:
            self._value = value * weight
        else:
            self._value = value
        self.subshape = value.shape[1:]
        self.norm = 1.0
    def num(self, index):
        return self._value[index]
    def denom(self, index):
        if self._weight is None:
            return numpy.ones([len(index)] + list(self.subshape))
        else:
            return self._weight[index]

class Bins(object):
    def __init__(self, *args):
        """ the shape has one extra per edge
            0 is .le. min
            -1 is .g. max
            args are (min, max, Nbins)
            the first is for R
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
                x = numpy.log10(args[d] / self.min[d])
            else:
                x = args[d] - self.min[d]
            integer[d] = numpy.ceil(args[d] * self.inv[d])
        return numpy.ravel_multi_index(integer, self.shape, mode='clip')

    def __call__(self, r, i, j):
        raise UnimplementedError()

class RmuBins(Bins):
    def __init__(self, Rmax, Nbins, Nmubins, observer):
        Bins.__init__(self, 
                (0, Rmax, Nbins),
                (0, 1, Nmubins)
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
        return self.linear(r, numpy.abs(mu))

class XYBins(Bins):
    """ the bins will be 
        [sky, los]
        with numpy imshow , the second axis los, will be vertical
                with imshow( ..T,) the sky will be vertical.
        """

    def __init__(self, Rmax, Nbins, observer):
        self.Rmax = Rmax
        Bins.__init__(self,
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

class RBins(Bins):
    def __init__(self, Rmax, Nbins, logscale=False, Rmin=0):
        Bins.__init__(self, 
                (Rmin, Rmax, Nbins, logscale))
    def __call__(self, r, i, j, data1, data2):
        return self.linear(r)

def paircount(data1, data2, bins, np=None):
    """ 
        returns bincenter, counts
        bins is instance of Bins, (eg, RBins, RmuBins)

        if the weight/value has multiple components,
        return counts with be 'tuple', one item for each component

        each count item has 2 more items than Nbins. 

        example 1:

        three cases:

        points x points
           returns weight1 * weight2 sum in bins
           normalized by weight1.sum() * weight2.sum()
        field x field
           returns the weighted mean of value1 * value2 in bins
             value1 * weight1 * value2 * weight2 / weight1 * weight2 
        field x point
           returns weighted mean of value1 * weight2 in bins:
             value1 * weight1 * weight2 / weight1
           normalized by weight2.sum()
        build data with point(pos, weight) or field(pos, value, weight).
        weight can be none.
        returns Rbins and the pair counts.
        For f-f, this is already the correlation function.
        for p-p, need to use landy-salay (whatever (DD - DR + RR) / RR
        for f-p, need to use O'Connell's estimator. haven't figure out yet.
    """
    tree1 = data1.tree
    tree2 = data2.tree
    p = list(kdcount.divide_and_conquer(tree1, tree2, 50000))

    fullshape = list(data1.subshape) + list(bins.shape)
    linearshape = [-1] + list(bins.shape)

    denomsum = numpy.zeros(fullshape, dtype='f8').reshape(linearshape)
    numsum = numpy.zeros(fullshape, dtype='f8').reshape(linearshape)

    def work(i):
        n1, n2 = p[i]
        num = numpy.zeros(fullshape, dtype='f8').reshape(linearshape)
        denom = numpy.zeros(fullshape, dtype='f8').reshape(linearshape)
        for r, i, j in n1.enumiter(n2, bins.Rmax):
            dig = bins(r, i, j, data1, data2)
            numij = data1.num(i) * data2.num(j)
            numij = numij.reshape(dig.size, -1)
            for d in range(num.shape[0]):
                num[d].flat [:] += numpy.bincount(dig, 
                        numij[:, d],
                        minlength=num[d].size)
            if isinstance(data1, field) \
            or isinstance(data2, field):
                denomij = data1.denom(i) * data2.denom(j)
                denomij = denomij.reshape(dig.size, -1)
                for d in range(num.shape[0]):
                    denom[d].flat [:] += numpy.bincount(dig,
                            denomij[:, d],
                        minlength=denom[d].size)
        return num, denom
    def reduce(num, denom):
        numsum[...] += num
        denomsum[...] += denom
    with Pool(use_threads=False, np=np) as pool:
        pool.map(work, range(len(p)), reduce=reduce)

    if isinstance(data1, points) \
        and isinstance(data2, points):
        # special handling for two point sets where
        # the denominator is always 1
        denomsum[...] = 1.0
    # this works no matter data.norm is scalar or vector.
    corr = (numsum / denomsum).T / (data1.norm * data2.norm)
    corr = corr.T.reshape(fullshape)
    return corr, bins

def main():
    pm = numpy.fromfile('A00_hodfit.raw').reshape(-1, 8)[::1, :3]
    wm = numpy.ones((len(pm), 2))
    martin = points(pm, wm)
    pr = numpy.random.uniform(size=(1000000, 3))
    wr = numpy.ones((len(pr), 2))
    random = points(pr, wr)
    DR, bins = paircount(martin, random, RBins(0.1, 40))
    DD, junk = paircount(martin, martin, RBins(0.1, 40))
    RR, junk = paircount(random, random, RBins(0.1, 40))
    return bins.centers, DD[...,1:-1], DR[..., 1:-1], RR[..., 1:-1]

def main2():
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    print 'read'
    print sim
    print sim.mean(dtype='f8'), sim.max()
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.4
    value = numpy.tile(sim[sample], (2, 1)).T
#    value = sim[sample]
    data = field(pos[sample], value=value)
    print 'data ready'
    DD = paircount(data, data, RBins(0.1, 40))
    return DD

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
    DD = paircount(data, data, RmuBins(0.10, 8, 20, 0.5))
    return DD
