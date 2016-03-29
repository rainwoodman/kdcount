from . import models
from .correlate import RBinning
import numpy

class points(models.points):
    def __init__(self, ra, dec, weights=None, boxsize=None, extra={}):
        ra = ra * (numpy.pi / 180.)
        dec = dec * (numpy.pi / 180.)
        pos = numpy.empty(len(ra), dtype=(ra.dtype, 3))
        pos[:, 2] = numpy.sin(dec)
        r = numpy.cos(dec)
        pos[:, 0] = numpy.sin(ra) * r
        pos[:, 1] = numpy.cos(ra) * r 

        models.points.__init__(self, pos, weights, boxsize, extra)

class AngularBinning(RBinning):
    def __init__(self, angbins, **kwargs):
        rbins = 2 * numpy.sin(0.5 * numpy.radians(angbins))
        RBinning.__init__(self, rbins, **kwargs)
    @property
    def angular_centers(self):
        return 2 * numpy.arcsin(self.centers * 0.5) * (180. / numpy.pi)
    @property
    def angular_edges(self):
        return 2 * numpy.arcsin(self.edges * 0.5) * (180. / numpy.pi)
