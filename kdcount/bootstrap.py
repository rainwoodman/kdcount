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
    def __init__(self, cache, sizes_per_var):
        self.cache = cache
        self.sizes_per_var = sizes_per_var

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

        self.active_straps = active_straps
        self.sizes = N
        self.size = N.sum()

    def bootstrap(self, straps=None, rng=None):
        """ create a bootstrap (of strap ids), that goes to self.resample.

            if straps is given only use those straps.
        """
        if rng is None:
            rng = numpy.random
        p = 1.0 * self.sizes / self.size
        if straps is None:
            straps = self.active_straps

        size = self.sizes[straps].sum()

        def inner():
            Nremain = size
            # XXX: is this fair?
            # XXX: the distribution of the total length is not well understood
            #     but mean is OK.
            while Nremain > 0:
                ch = rng.choice(len(straps), size=1, replace=True)
                sid = straps[ch]
                accept = rng.uniform() <= Nremain / (1.0 * self.sizes[sid])
                if accept:
                    Nremain -= self.sizes[sid]
                    yield sid
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
        varstraps = [self.create_straps(v) for v in args]
        indstraps = [[int(i) for i in  self.active_straps]]*len(args)
        cache = {}
        sizes_per_var = [{} for i in range(len(args))]

        with utils.MapReduce(np=np) as pool:
            def work(p):
                ind, vars = p
                n = [len(var) for var in vars]
                return ind, estimator(*vars), n
            def reduce(ind, r, n):
                cache[ind] = r
                for i, (ind1, n1) in enumerate(zip(ind, n)):
                    sizes_per_var[i][ind1] = n1

            items = [(ind, var) for ind, var in zip(outer(*indstraps), outer(*varstraps))]
            pool.map(work, items, reduce=reduce)

        result = StrappedResults(cache, sizes_per_var)
        return result

    def resample(self, result, bootstrap=None, operator=lambda x, y : x + y):
        """
            bootstrap is a list of strapids returnedy by self.bootstrap.
            operator is used to combine the result of straps into the resample.
            it shall be sort of addition but may have to be different if result
            of the function does not support the `+` operator.

        """
        Nargs = len(result.sizes_per_var)
        # length for each bootstrapped resample dataset
        L = [sum([sizes_per_strap[i] for i in bootstrap]) for sizes_per_strap in result.sizes_per_var]
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
