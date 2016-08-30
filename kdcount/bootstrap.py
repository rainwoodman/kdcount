"""

    Bootstrap / Bagging.

    Data Model:
        dataset is chopped to active_straps by bsfun: dataset -> id

        bag produces strap ids.

    See the documentation of :class:`policy`

"""
from itertools import product as outer
from functools import reduce
from .models import points
from .correlate import paircount
from . import utils 
import numpy

class StrappedResults(object):
    def __init__(self, cache, sizes):
        self.cache = cache
        self.sizes = sizes

class policy(object):
    """
        Bootstrap policy for bootstrapping a multi-distributive estimator of data.

        multi-distributive estimator satisfies

        .. math :

            E(x + y, \dots)= E(x, dots) + E(y, \dots)

        where :math:`x` :math:`y` are datasets and the sum is extending dataset.
        :math:`E(\odot)` is the estimator. 

        If an estimator is distributive, bootstraps can be calculated efficiently by combining
        precomputed estimators of subsamples (straps). [ e.g. Martin White's wcross.cpp ] 
        This acceleration is implemented here.

        Fun Fact
        --------
        The term was coined at AstroHackWeek2016 (astrohackweek.org) by
        @rainwoodman and @drphilmarshall, after ruling out
        a few wrongly-suggestive alternatives.
        (linear, commutitave, or 'Bienaymial'). We yet have
        to see if statistians have a better name for this attribute.

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
        """ create a bootstrap (of internal strap ids), that goes to self.resample"""
        if rng is None:
            rng = numpy.random
        p = 1.0 * self.sizes / self.size
        def inner():
            Nremain = self.size
            # XXX: is this fair?
            # XXX: the distribution of the total length is not well understood
            #     but mean is OK.
            while Nremain > 0:
                ch = rng.choice(len(self.active_straps), size=1, replace=True)
                accept = rng.uniform() <= Nremain / (1.0 * self.sizes[ch])
                if accept:
                    Nremain -= self.sizes[ch]
                    yield ch
                else:
                    # in this case Nremain is less than the size of the chosen chunk.
                    break
        return numpy.fromiter(inner(), dtype='i4')

    def run(self, estimator, *args, **kwargs):
        """
            run estimator on the 'straps'

            Parameters
            ----------
            *args : a list of indexable datasets.
                   the datasets are divided into straps by the strap function.
            estimator: the estimator `estimator(*args)`.

            np : number of processes to use
        """
        np = kwargs.pop('np', None)
        vars = [self.create_straps(v) for v in args]
        cache = {}

        with utils.MapReduce(np=np) as pool:
            def work(p):
                ind, var = p
                return ind, estimator(*var)
            def reduce(ind, r):
                cache[ind] = r

            items = [(ind, var) for ind, var in zip(outer(*([range(len(self.active_straps))]*len(args))),
                                outer(*vars))]
            pool.map(work, items, reduce=reduce)

        sizes = [numpy.array([len(s) for s in v], dtype='intp') for v in vars]
        result = StrappedResults(cache, sizes)
        return result

    def resample(self, result, bootstrap=None, operator=lambda x, y : x + y):
        """
            bootstrap is a list of strapids returnedy by self.bootstrap.
            operator is used to combine the result of straps into the resample.
            it shall be sort of addition but may have to be different if result
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
