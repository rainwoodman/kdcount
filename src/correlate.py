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
    def __init__(self, pos):
        self.pos = pos
        self.tree = kdcount.build(self.pos)

class points(dataset):
    def __init__(self, pos, weight=None):
        dataset.__init__(self, pos)
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
    def __init__(self, pos, value, weight=None):
        dataset.__init__(self, pos)
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
    def __init__(self, Rmax, edges):
        self.minlength = -1
        self.Rmax = Rmax
        self.edges = edges
        if isinstance(edges, tuple):
            self.shape = tuple([e.shape[0] for e in edges])
        else:
            self.shape = edges.shape

    def __call__(self, r, i, j):
        raise UnimplementedError()

class RmuBins(Bins):
    def __init__(self, Rmax, Nbins, Nmubins, observer):
        self.Rmax = Rmax
        Bins.__init__(self, Rmax, 
            (
            numpy.arange(Nbins + 1) * (1.0 * Rmax) / Nbins,
            numpy.arange(Nmubins + 1) * (1.0) / Nmubins
            )
            )
        self.invdR = Nbins / (1.0 * Rmax)
        self.invdmu = Nmubins / 1.0
        self.observer = numpy.array(observer)

    def __call__(self, r, i, j, data1, data2):
        Nbins = self.shape[0] - 1
        x = numpy.int32(r * self.invdR).clip(0, Nbins)

        Nmubins = self.shape[1] - 1
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        center = numpy.einsum('ij, ij->i', center, center) ** 0.5
        mu = dot / (center * r)
        mu[r == 0] = 10.0
        y = numpy.int32(numpy.abs(mu) * self.invdmu).clip(0, Nmubins)

        return numpy.ravel_multi_index((x, y), self.shape)

class XYBins(Bins):
    def __init__(self, Rmax, Nbins, observer):
        self.Rmax = Rmax
        Bins.__init__(self, Rmax, 
            (
            numpy.arange(Nbins + 1) * (1.0 * Rmax) / Nbins,
            numpy.arange(Nbins + 1) * (1.0 * Rmax) / Nbins,
            )
            )
        self.invdR = Nbins / (1.0 * Rmax)
        self.observer = observer

    def __call__(self, r, i, j, data1, data2):
        Nbins = self.shape[0] - 1
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        dot2 = dot ** 2
        center2 = numpy.einsum('ij, ij->i', center, center)
        y2 = dot2 / center2
        dr2 = numpy.einsum('ij, ij->i', dr, dr)
        x2 = numpy.abs(dr2 - y2)
        x = numpy.int32(x2 ** 0.5 * self.invdR).clip(0, Nbins)
        y = numpy.int32(y2 ** 0.5 * self.invdR).clip(0, Nbins)
        return numpy.ravel_multi_index((x, y), self.shape)

class RBins(Bins):
    def __init__(self, Rmax, Nbins):
        self.Rmax = Rmax
        Bins.__init__(self, Rmax, 
            numpy.arange(Nbins + 1) * (1.0 * Rmax) / Nbins
        )
        self.invdR = Nbins / (1.0 * Rmax)
    def __call__(self, r, i, j, data1, data2):
        bins = self.shape[0] - 1
        return numpy.int16(r * self.invdR).clip(0, bins)

def paircount(data1, data2, bins):
    """ 
        returns bincenter, counts
        bins is instance of Bins, (RBins, RmuBins)

        if the weight/value has multiple components,
        the counts of each component is returned as columns

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
    with Pool(use_threads=False) as pool:
        pool.map(work, range(len(p)), reduce=reduce)

    if isinstance(data1, points) \
        and isinstance(data2, points):
        # special handling for two point sets where
        # the denominator is always 1
        denomsum[...] = 1.0
    corr = (numsum / denomsum).T / (data1.norm * data2.norm)
    corr = corr.T.reshape(fullshape).T
    return bins.edges, corr

def main():
    pm = numpy.fromfile('A00_hodfit.raw').reshape(-1, 8)[::1, :3]
    wm = numpy.ones((len(pm), 2))
    martin = points(pm, wm)
    pr = numpy.random.uniform(size=(1000000, 3))
    wr = numpy.ones((len(pr), 2))
    random = points(pr, wr)
    DR = paircount(martin, random, RBins(0.1, 40))
    DD = paircount(martin, martin, RBins(0.1, 40))
    RR = paircount(random, random, RBins(0.1, 40))
    return DR[0], DD[1], DR[1], RR[1]

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
