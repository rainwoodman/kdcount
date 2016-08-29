"""

    Bootstrap / Bagging.

    Data Model:
        dataset is chopped to active_straps by bsfun: dataset -> id

        bag produces strap ids.

"""
from itertools import product as outer
from .models import points
from .correlate import paircount
import numpy

class policy(object):
    """
        Attributes
        ----------
        active_straps : ID of straps that are non-empty.
        sizes   : sizes of active_straps, same length of active_straps.

    """
    def __init__(self, strapfunc, dataset):
        """ A bag object that produces equal lengthed samples from bsfun.

            Dataset provides the barebone of the policy.
            We aim each bootstrap resample to have the same
            size as of the dataset.
        """
        self.strapfunc = strapfunc
        sid = strapfunc(dataset)
        N = numpy.bincount(sid)
        active_straps = N.nonzero()[0]
        N = N[active_straps]

        self.active_straps = active_straps
        self.sizes = N
        self.size = N.sum()

    def resample(self, rng=None):
        """ create a resample (of strap ids)"""
        if rng is None:
            rng = numpy.random
        p = 1.0 * self.sizes / self.size
        def inner():
            Nremain = self.size
            # XXX: is this fair?
            while Nremain > 0:
                ch = rng.choice(self.active_straps, size=1, replace=True, p=p)
                accept = rng.uniform() <= Nremain / (1.0 * self.sizes[ch])
                if accept:
                    Nremain -= self.sizes[ch]
                    yield ch
                else:
                    # in this case Nremain is less than the size of the chosen chunk.
                    break
        return numpy.fromiter(inner(), dtype=self.active_straps.dtype)

    def run(self, func, *args):
        vars = [self.create_straps(v) for v in args]
        result = lambda : None
        result.cache = {}
        for ind, var in zip(outer(*([self.active_straps]*len(args))),
                            outer(*vars)):
            result.cache[ind] = func(*var)
        result.sizes = [numpy.array([len(s) for s in v], dtype='intp') for v in vars]
        return result

    def create_resample(self, result, strapids):
        Nargs = len(result.sizes)
        # length for each bootstrapped resample dataset
        L = [sum(s[strapids]) for s in result.sizes]
        # result is the sum of the submatrix
        R = sum([result.cache[ind] for ind in outer(*([strapids] * Nargs))])
        return L, R

    def create_straps(self, data):
        index = self.strapfunc(data)

        a = numpy.argsort(index)
        N = numpy.bincount(index, minlength=self.active_straps.max() + 1)

        end = N.cumsum()
        start = end.copy()
        start[1:] = end[:-1]
        start[0] = 0

        return [data[a[slice(*i)]] for i in zip(start[self.active_straps], end[self.active_straps])]
