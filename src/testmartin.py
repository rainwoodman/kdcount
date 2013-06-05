import pyximport
import numpy
import cython
import sharedmem

pyximport.install(setup_args={'include_dirs': [numpy.get_include(), './']})
import pykdcount

noise = numpy.random.random(size=(1000000, 3))
martin = numpy.loadtxt('A00_hodfit.gal', usecols=(0, 1, 2))

R = pykdcount.build(noise, 1.0)
D = pykdcount.build(martin, 1.0)

xbins = numpy.linspace(-0.125, 0.125, 128, endpoint=True)
ybins = numpy.linspace(0, 0.125, 64, endpoint=True)
rbins = numpy.linspace(0, 0.125, 64, endpoint=True)
mubins = numpy.linspace(0, 1, 64, endpoint=False)

def fastdig(data, bins):
    idx = 1.0 / (bins[1] - bins[0])
    id = numpy.intp((data - bins[0]) * idx) + 1
    id.clip(0, bins.size, out=id)
    return id

RR = numpy.zeros((xbins.size + 1, ybins.size + 1) , dtype='i8')
RD = numpy.zeros((xbins.size + 1, ybins.size + 1) , dtype='i8')
DD = numpy.zeros((xbins.size + 1, ybins.size + 1) , dtype='i8')
RRr = numpy.zeros((rbins.size + 1, mubins.size + 1), dtype='i8')
RDr = numpy.zeros((rbins.size + 1, mubins.size + 1), dtype='i8')
DDr = numpy.zeros((rbins.size + 1, mubins.size + 1), dtype='i8')

def eval_mu(x, y, r, observer = (2070. / 1500.) - 0.5):
    dx = y - x
    dx[dx < -0.5] += 1.0
    dx[dx > 0.5] -= 1.0
    mid = dx.copy()
    mid *= 0.5
    mid += x
    mid[:, 2] += observer
    dot = numpy.einsum('ij,ij->i', mid, dx)
    dist = numpy.einsum('ij,ij->i', mid, mid) ** 0.5
    dist *= r
    mu = dot / dist
    return mu
def process(r, i, j, A, B, target, targetr):
    good = r != 0
    i = i[good]
    j = j[good]
    r = r[good]

    x = A[i]
    y = B[j]
    mu = eval_mu(x, y, r)
    dx = r * mu
    dy = r * (1 - mu * mu) ** 0.5

    xdig = fastdig(dx, xbins)
    ydig = fastdig(dy, ybins)
    rdig = fastdig(r, rbins)
    mudig = fastdig(numpy.abs(mu), mubins)
    dig = numpy.ravel_multi_index((xdig, ydig), RR.shape)
    target[:] += numpy.bincount(dig, minlength=target.size)
    dig = numpy.ravel_multi_index((rdig, mudig), RRr.shape)
    targetr[:] += numpy.bincount(dig, minlength=targetr.size)

def paracount(node1, node2, A, B, target, targetr):
    with sharedmem.Pool() as pool:
        def work(other):
            tmp = numpy.zeros_like(target)
            tmpr = numpy.zeros_like(targetr)
            for r, i, j in node1.enum(other, 0.125, bunch=128 * 1024):
                process(r, i, j, A, B, target=tmp, targetr=tmpr)
            return tmp, tmpr
        def reduce(tmp, tmpr):
            target[:] += tmp
            targetr[:] += tmpr
        pool.map(work, 
                node2.subtrees(node2.size // 512), reduce=reduce)
print 'started'
paracount(R, R, noise, noise, RR.ravel(), RRr.ravel())
print 'RR'
paracount(R, D, noise, martin, RD.ravel(), RDr.ravel())
print 'RD'
paracount(D, D, martin, martin, DD.ravel(), DDr.ravel())
print 'DD'

fac = D.size * 1.0 / R.size
xi = (RR * fac ** 2 - 2 * RD * fac + DD) / (RR * fac ** 2)
xir = (RRr * fac ** 2 - 2 * RDr * fac + DDr) / (RRr * fac ** 2)


