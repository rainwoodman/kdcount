import kdcount
import numpy

from kdcount import sphere, correlate
from kdcount import bootstrap
from numpy.testing import assert_allclose

def test_bootstrap():
    numpy.random.seed(1234)
    ra, dec = numpy.random.uniform(size=(2, 100000))
    ra = ra * 360
    dec = numpy.arcsin((dec - 0.5) * 2) / numpy.pi * 180
    Nbar = len(ra) / (4. * numpy.pi * (180 / numpy.pi) ** 2)

    ds = sphere.points(ra, dec)

    bs = lambda x: sphere.radec2pix(2, x.ra, x.dec)
    bpc = bootstrap.bpaircount(ds, ds, sphere.AngularBinning(numpy.linspace(0, 1, 10)), bootstrapper=bs, np=0)

    pc = correlate.paircount(ds, ds, sphere.AngularBinning(numpy.linspace(0, 1, 10)), np=0)

    assert_allclose(pc.weight, bpc.weight, rtol=1e-3)
    assert_allclose(pc.sum1, bpc.sum1, rtol=1e-3)
