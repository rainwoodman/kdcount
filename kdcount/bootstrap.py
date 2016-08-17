from itertools import product as outer
from .models import points
from .correlate import paircount
import numpy

def binslices(index):
    a = numpy.argsort(index)
    N = numpy.bincount(index)

    end = N.cumsum()
    start = end.copy()
    start[1:] = end[:-1]
    start[0] = 0

    return [slice(*i) for i in zip(start, end)]

class bpaircount(object):
    """
        Parameters
        ----------
        bootstrapper: callable(dataset) -> integer array
           mapping dataset to integer subsample id. The subsamples are assembled into
            bootstrap samples. Currently we remove 1 subsample to form a bootstrap sample.
            Is this called jackknife?

        other parameters are passed to paircount

        Returns
        -------
        an bpaircount object . Attributes are:

        sum1, sum2, weight: as paircount

        samplesum1, samplesum2: lists of sum1, sum2, per bootstrap sample.
        sampleweight : the total weight (product of norm or number of points) per sample
        
        ddof : delta degrees of freedom in the bootstrap samples. It is close to but not -2.
            due to the way subsamples are created.

    """
    def __init__(self, data1, data2, binning, bootstrapper, usefast=True, np=None):
        pts_only = isinstance(data1, points) and isinstance(data2, points)

        junk, bnshape = binning.sum_shapes(data1, data2)

        # chop off the data
        s1 = binslices(bootstrapper(data1))
        s2 = binslices(bootstrapper(data2))

        # match the length
        if len(s1) < len(s2):
            s1.extend([slice(0, 0)] * (len(s2) - len(s1)))
        if len(s2) < len(s1):
            s2.extend([slice(0, 0)] * (len(s1) - len(s2)))

        data1 = [ data1[s] for s in s1]
        data2 = [ data2[s] for s in s2]

        bsshape = (len(data1), len(data2))

        self.bsweights = numpy.zeros(bsshape, ('f8'))
        self.bsfullsum1 = numpy.zeros(bsshape, ('f8', bnshape))
        if not pts_only:
            self.bsfullsum2 = numpy.zeros(bsshape, ('f8', bnshape))

        for i, j in numpy.ndindex(*bsshape):
            d1, d2 = data1[i], data2[j]
            if len(d1) > 0 and len(d2) > 0:
                pc = paircount(d1, d2, binning, usefast, np)

                self.bsfullsum1[i, j] = pc.fullsum1
                if not pts_only:
                    self.bsfullsum2[i, j] = pc.fullsum2
                self.bsweights[i, j] = 1.0 * data1[i].norm * data2[j].norm
            
        self.edges = binning.edges
        self.centers = binning.centers

        # make samples
        Nsamples = len(data1)
        self.samplefullsum1 = numpy.zeros(Nsamples, ('f8', bnshape))
        if not pts_only:
            self.samplefullsum2 = numpy.zeros(Nsamples, ('f8', bnshape))

        self.sampleweight = numpy.zeros(Nsamples, ('f8', [1] * len(bnshape)))

        for i in range(Nsamples):
            mask = numpy.ones(list(bsshape) + [1] * len(bnshape))
            mask[i, :] = 0
            mask[:, i] = 0
            self.samplefullsum1[i] = (self.bsfullsum1 * mask).sum(axis=(0, 1))
            if not pts_only:
                self.samplefullsum2[i] = (self.bsfullsum2 * mask).sum(axis=(0, 1))

            self.sampleweight[i] = (self.bsweights * mask).sum()

        # sample mean
        # I fiddled and -2 + 1.0 / Nsamples gives 'roughly' the correct number.
        # If we look at the sum of bsweights and wampleweight, the ratio is Nsamples - 2 + 1.0 / Nsamples
        ndof = (Nsamples - 2 + 1.0 / Nsamples)
        self.weight = self.sampleweight.sum() / ndof

        # I don't really know if we shall weight by the sampels or not.
        self.fullsum1 = (self.samplefullsum1 * self.sampleweight).sum(axis=0) / self.sampleweight.mean() / ndof
        self.sum1 = self.fullsum1[[Ellipsis] + [slice(1, -1)] * binning.Ndim]
        self.samplesum1 = self.samplefullsum1[[Ellipsis] + [slice(1, -1)] * binning.Ndim]

        self.ndof = ndof
        self.ddof = ndof - Nsample
        if not pts_only:
            self.fullsum2 = (self.samplefullsum2 * self.sampleweight).sum(axis=0) / self.sampleweight.mean() / ndof
            self.sum2 = self.fullsum2[[Ellipsis] + [slice(1, -1)] * binning.Ndim]
            self.samplesum2 = self.samplefullsum2[[Ellipsis] + [slice(1, -1)] * binning.Ndim]

