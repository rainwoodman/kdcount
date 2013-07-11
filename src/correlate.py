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
            self.norm = weight.sum()
        else:
            self.norm = len(pos) * 1.0
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
        self.norm = 1.0
    def num(self, index):
        return self._value[index]
    def denom(self, index):
        if self._weight is None:
            return numpy.ones(len(index))
        else:
            return self._weight[index]

class digitize(object):
    def __init__(self):
        self.minlength = -1
    def __call__(self, r, i, j):
        raise UnimplementedError()

class digitizeRmu2(digitize):
    def __init__(self, Rmax, Nbins, Nmu2bins, observer):
        digitize.__init__(self)
        self.invdR = Nbins / (1.0 * Rmax)
        self.invdmu2 = Nmu2bins / 1.0
        self.bins = (
                numpy.arange(Nbins + 1) * (1.0 * Rmax) / Nbins,
                numpy.arange(Nmu2bins + 1) * (1.0) / Nmu2bins)
        self.shape = (Nbins + 1, Nmu2bins + 1)
        self.Rmax = Rmax
        self.observer = observer

    def __call__(self, r, i, j, data1, data2):
        Nbins = self.shape[0] - 1
        Nmu2bins = self.shape[1] - 1
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij-> i', dr, center) 
        center = numpy.einsum('ij, ij->i', center, center) ** 0.5
        mask = (r != 0) & (center != 0)
        dot[mask] /= (r[mask] * center[mask])
        dot[~mask] = 10
        mu2 = dot * dot
        x = numpy.int32(r * self.invdR).clip(0, Nbins)
        y = numpy.int32(mu2 * self.invdmu2).clip(0, Nmu2bins)
        return numpy.ravel_multi_index((x, y), self.shape)

class digitizeR(digitize):
    def __init__(self, Rmax, Nbins):
        digitize.__init__(self)
        self.invdR = Nbins / (1.0 * Rmax)
        self.bins = numpy.arange(Nbins + 1) * (1.0 * Rmax) / Nbins
        self.shape = Nbins + 1
        self.Rmax = Rmax
    def __call__(self, r, i, j, data1, data2):
        bins = self.shape - 1
        return numpy.int16(r * self.invdR).clip(0, bins)

def paircount(data1, data2, d):
    """ 
        d is instance of digitize, ( digitizeR)

        example 1:

    martin = points(numpy.fromfile('A00_hodfit.raw').reshape(-1, 8)[::1, :3])
    random = points(numpy.random.uniform(size=(1000000, 3)))
    DR = paircount(martin, random, digitizeR(0.1, 40))
    DD = paircount(martin, martin, digitizeR(0.1, 40))
    RR = paircount(random, random, digitizeR(0.1, 40))

    example 2
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.4
    data = field(pos[sample], value=sim[sample])
    print 'data ready'
    DD = paircount(data, data, digitizeR(0.1, 40))
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

    denomsum = numpy.zeros(d.shape, dtype='f8')
    numsum = numpy.zeros(d.shape, dtype='f8')

    def work(i):
        n1, n2 = p[i]
        num = numpy.zeros(d.shape, dtype='f8')
        denom = numpy.zeros(d.shape, dtype='f8')
        for r, i, j in n1.enumiter(n2, d.Rmax):
            dig = d(r, i, j, data1, data2)
            num.flat [:] += numpy.bincount(dig, 
                    data1.num(i) * data2.num(j), 
                    minlength=num.size)
            if isinstance(data1, field) \
            or isinstance(data2, field):
                denom.flat [:] += numpy.bincount(dig,
                    data1.denom(i) * data2.denom(j),
                    minlength=denom.size)
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
    corr = numsum / denomsum / (data1.norm * data2.norm)
    return d.bins, corr

def main():
    martin = points(numpy.fromfile('A00_hodfit.raw').reshape(-1, 8)[::1, :3])
    random = points(numpy.random.uniform(size=(1000000, 3)))
    DR = paircount(martin, random, digitizeR(0.1, 40))
    DD = paircount(martin, martin, digitizeR(0.1, 40))
    RR = paircount(random, random, digitizeR(0.1, 40))
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
    data = field(pos[sample], value=sim[sample])
    print 'data ready'
    DD = paircount(data, data, digitizeR(0.1, 40))
    return DD

def main3():
    sim = numpy.fromfile('grid-128.raw', dtype='f4')
    print 'read'
    pos = numpy.array(numpy.unravel_index(numpy.arange(sim.size),
        (128, 128, 128))).T / 128.0
    numpy.random.seed(1000)
    sample = numpy.random.uniform(size=len(pos)) < 0.2
    data = field(pos[sample], value=sim[sample])
    print 'data ready'
    DD = paircount(data, data, digitizeRmu2(0.10, 8, 20, 0.5))
    return DD
