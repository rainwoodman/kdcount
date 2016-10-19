from . import models
from .correlate import RBinning
import numpy

class points(models.points):
    def __init__(self, ra, dec, weights=None, boxsize=None):
        self.ra = ra
        self.dec = dec
        ra = ra * (numpy.pi / 180.)
        dec = dec * (numpy.pi / 180.)
        dtype = numpy.dtype((ra.dtype, 3))
        pos = numpy.empty(len(ra), dtype=dtype)
        pos[:, 2] = numpy.sin(dec)
        r = numpy.cos(dec)
        pos[:, 0] = numpy.sin(ra) * r
        pos[:, 1] = numpy.cos(ra) * r 

        models.points.__init__(self, pos, weights, boxsize)

    def __getitem__(self, index):
        return points(self.ra[index], self.dec[index], self.weights[index], self.boxsize);

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

import heapq
def bootstrap(nside, rand, nbar, *data):
    """ This function will bootstrap data based on the sky coverage of rand.
        It is different from bootstrap in the traditional sense, but for correlation
        functions it gives the correct answer with less computation.

        nbar : number density of rand, used to estimate the effective area of a pixel

        nside : number of healpix pixels per side to use

        *data : a list of data -- will be binned on the same regions.

        small regions (incomplete pixels) are combined such that the total
        area is about the same (a healpix pixel) in each returned boot strap sample

        Yields: area, random, *data

        rand and *data are in (RA, DEC)

        Example:

        >>> for area, ran, data1, data2 in bootstrap(4, ran, 100., data1, data2):
        >>>    # Do stuff
        >>>    pass
    """

    def split(data, indices, axis):
        """ This function splits array. It fixes the bug 
            in numpy that zero length array are improperly handled.

            In the future this will be fixed.
        """
        s = []
        s.append(slice(0, indices[0]))
        for i in range(len(indices) - 1):
            s.append(slice(indices[i], indices[i+1]))
        s.append(slice(indices[-1], None))

        rt = []
        for ss in s:
            ind = [slice(None, None, None) for i in range(len(data.shape))]
            ind[axis] = ss
            ind = tuple(ind)
            rt.append(data[ind])
        return rt

    def hpsplit(nside, data):
        # data is (RA, DEC)
        RA, DEC = data
        pix = radec2pix(nside, RA, DEC)
        n = numpy.bincount(pix)
        a = numpy.argsort(pix)
        data = numpy.array(data)[:, a]
        rt = split(data, n.cumsum(), axis=-1)
        return rt

    # mean area of sky.

    Abar =  41252.96 / nside2npix(nside)
    rand = hpsplit(nside, rand)
    if len(data) > 0:
        data = [list(i) for i in zip(*[hpsplit(nside, d1) for d1 in data])]
    else:
        data = [[] for i in range(len(rand))]

    heap = []
    j = 0
    for r, d in zip(rand, data):
        if len(r[0]) == 0: continue
        a = 1.0 * len(r[0]) / nbar
        j = j + 1
        if len(heap) == 0:
            heapq.heappush(heap, (a, j, r, d))
        else:
            a0, j0, r0, d0 = heapq.heappop(heap)
            if a0 + a < Abar:
                a0 += a
                d0 = [
                     numpy.concatenate((d0[i], d[i]), axis=-1)
                     for i in range(len(d))
                    ]
                r0 = numpy.concatenate((r0, r), axis=-1)
            else:
                heapq.heappush(heap, (a, j, r, d))
            heapq.heappush(heap, (a0, j0, r0, d0))

    for i in range(len(heap)):
        area, j, r, d = heapq.heappop(heap)
        rt = [area, r] + d
        yield rt

def pix2radec(nside, pix):
    theta, phi = pix2ang(nside, pix)
    return numpy.degrees(phi), 90 - numpy.degrees(theta)

def radec2pix(nside, ra, dec):
    phi = numpy.radians(ra)
    theta = numpy.radians(90 - dec)
    return ang2pix(nside, theta, phi)
 
def nside2npix(nside):
    return nside * nside * 12

def ang2pix(nside, theta, phi):
    r"""Convert angle :math:`\theta` :math:`\phi` to pixel.

        This is translated from chealpix.c; but refer to Section 4.1 of
        http://adsabs.harvard.edu/abs/2005ApJ...622..759G
    """
    nside, theta, phi = numpy.lib.stride_tricks.broadcast_arrays(nside, theta, phi)
    
    def equatorial(nside, tt, z):
        t1 = nside * (0.5 + tt)
        t2 = nside * z * 0.75
        jp = (t1 - t2).astype('i8')
        jm = (t1 + t2).astype('i8')
        ir = nside + 1 + jp - jm # in {1, 2n + 1}
        kshift = 1 - (ir & 1) # kshift=1 if ir even, 0 odd 
 
        ip = (jp + jm - nside + kshift + 1) // 2 # in {0, 4n - 1}
        
        ip = ip % (4 * nside)
        return nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip
        
    def polecaps(nside, tt, z, s):
        tp = tt - numpy.floor(tt)
        za = numpy.abs(z)
        tmp = nside * s / ((1 + za) / 3) ** 0.5
        mp = za > 0.99
        tmp[mp] = nside[mp] * (3 *(1-za[mp])) ** 0.5
        jp = (tp * tmp).astype('i8')
        jm = ((1 - tp) * tmp).astype('i8')
        ir = jp + jm + 1
        ip = (tt * ir).astype('i8')
        ip = ip % (4 * ir)

        r1 = 2 * ir * (ir - 1) 
        r2 = 2 * ir * (ir + 1)
 
        r = numpy.empty_like(r1)
        
        r[z > 0] = r1[z > 0] + ip[z > 0]
        r[z < 0] = 12 * nside[z < 0] * nside[z < 0] - r2[z < 0] + ip[z < 0]
        return r
    
    z = numpy.cos(theta)
    s = numpy.sin(theta)
    
    tt = (phi / (0.5 * numpy.pi) ) % 4 # in [0, 4]
    
    result = numpy.zeros(z.shape, dtype='i8')
    mask = (z < 2. / 3) & (z > -2. / 3)
  
    result[mask] = equatorial(nside[mask], tt[mask], z[mask])
    result[~mask] = polecaps(nside[~mask], tt[~mask], z[~mask], s[~mask])
    return result
    
def pix2ang(nside, pix):
    r"""Convert pixel to angle :math:`\theta` :math:`\phi`.

        nside and pix are broadcast with numpy rules.

        Returns: theta, phi

        This is translated from chealpix.c; but refer to Section 4.1 of
        http://adsabs.harvard.edu/abs/2005ApJ...622..759G
    """
    nside, pix = numpy.lib.stride_tricks.broadcast_arrays(nside, pix)
    
    ncap = nside * (nside - 1) * 2
    npix = 12 * nside * nside
    
    def northpole(pix, npix):
        iring = (1 + ((1 + 2 * pix) ** 0.5)).astype('i8') // 2
        iphi = (pix + 1) - 2 * iring * (iring - 1)
        z = 1.0 - (iring*iring) * 4. / npix
        phi = (iphi - 0.5) * 0.5 * numpy.pi / iring
        return z, phi
    
    def equatorial(pix, nside, npix, ncap):
        ip = pix - ncap
        iring = ip // (4 * nside) + nside
        iphi = ip % (4 * nside) + 1
        fodd = (((iring + nside) &1) + 1.) * 0.5
        z = (2 * nside - iring) * nside * 8.0 / npix
        phi = (iphi - fodd) * (0.5 * numpy.pi) / nside
        return z, phi
    
    def southpole(pix, npix):
        ip = npix - pix
        iring = (1 + ((2 * ip - 1)**0.5).astype('i8')) // 2
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        z = -1 + (iring * iring) * 4. / npix
        phi = (iphi - 0.5 ) * 0.5 * numpy.pi / iring
        return z, phi
    
    mask1 = pix < ncap
    
    mask2 = (~mask1) & (pix < npix - ncap)
    mask3 = pix >= npix - ncap

    z = numpy.zeros(pix.shape, dtype='f8')
    phi = numpy.zeros(pix.shape, dtype='f8')
    z[mask1], phi[mask1] = northpole(pix[mask1], npix[mask1])
    z[mask2], phi[mask2] = equatorial(pix[mask2], nside[mask2], npix[mask2], ncap[mask2])
    z[mask3], phi[mask3] = southpole(pix[mask3], npix[mask3])
    return numpy.arccos(z), phi

