"""

    Bootstrap / Bagging.

    Data Model:
        dataset is chopped to active_straps by bsfun: dataset -> id

        bag produces strap ids.

"""
from itertools import product as outer
from functools import reduce
from .models import points
from .correlate import paircount
import numpy

class StrappedResults(object):
    def __init__(self, cache, sizes):
        self.cache = cache
        self.sizes = sizes

class policy(object):
    """
        Bootstrap policy for bootstrapping a multi-linear estimator of data.

        multi-linear estimator satisfies

        .. math :

            E(x + y) = E(x) + E(y)

        where :math:`x` :math:`y` are datasets and the sum is extending dataset.
        :math:`E(\odot)` is the estimator.

        Attributes
        ----------
        active_straps : ID of straps that are non-empty.
        sizes   : sizes of active_straps, same length of active_straps.

    """
    def __init__(self, strapfunc, dataset):
        """ A bag object that produces equal lengthed samples from bsfun.

            Dataset and strapfunc defines the policy.
            We aim each bootstrap resample to have the same
            size as of the original dataset.
        """
        self.strapfunc = strapfunc
        sid = strapfunc(dataset)
        N = numpy.bincount(sid)
        active_straps = N.nonzero()[0]
        N = N[active_straps]

        self.active_straps = active_straps
        self.sizes = N
        self.size = N.sum()

    def bootstrap(self, rng=None):
        """ create a bootstrap (of strap ids), that goes to self.resample"""
        if rng is None:
            rng = numpy.random
        p = 1.0 * self.sizes / self.size
        def inner():
            Nremain = self.size
            # XXX: is this fair?
            # XXX: the distribution of the total length is not well understood
            #     but mean is OK.
            while Nremain > 0:
                ch = rng.choice(self.active_straps, size=1, replace=True)
                accept = rng.uniform() <= Nremain / (1.0 * self.sizes[ch])
                if accept:
                    Nremain -= self.sizes[ch]
                    yield ch
                else:
                    # in this case Nremain is less than the size of the chosen chunk.
                    break
        return numpy.fromiter(inner(), dtype=self.active_straps.dtype)

    def run(self, estimator, *args):
        """
            run estimator on the 'straps'

            Parameters
            ----------
            *args : a list of indexable datasets.
                   the datasets are divided into straps by the strap function.
            estimator: the estimator `estimator(*args)`.

        """
        vars = [self.create_straps(v) for v in args]
        cache = {}
        for ind, var in zip(outer(*([self.active_straps]*len(args))),
                            outer(*vars)):
            cache[ind] = estimator(*var)
        sizes = [numpy.array([len(s) for s in v], dtype='intp') for v in vars]
        result = StrappedResults(cache, sizes)
        return result

    def resample(self, result, bootstrap=None, operator=lambda x, y : x + y):
        """
            bootstrap is a list of strapids returnedy by self.bootstrap.
            operator is used to combine the result of straps into the resample.
            it shall be sort of addition but may have to be different of result
            of the function does not support the `+` operator.

        """
        Nargs = len(result.sizes)
        # length for each bootstrapped resample dataset
        L = [sum(s[bootstrap]) for s in result.sizes]
        # result is the sum of the submatrix
        R = reduce(operator, [result.cache[ind] for ind in outer(*([bootstrap] * Nargs))])
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
