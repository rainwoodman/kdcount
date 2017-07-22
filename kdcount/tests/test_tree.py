from kdcount import KDTree, KDAttr
import numpy
from numpy.testing import assert_equal, run_module_suite, assert_array_equal
from kdcount.utils import constant_array
from kdcount.pykdcount import _get_last_fof_info

def test_build():
    numpy.random.seed(1000)
    pos = numpy.random.uniform(size=(1000, 3))
    tree = KDTree(pos).root
    N = [0, 0]
    def process1(r, i, j):
        N[0] += len(r)

    def process2(r, i, j):
        N[1] += len(r)

    tree.enum(tree, rmax=0.1, process=process1)

    #tree.enum_point(pos, rmax=0.1, process=process2)

    #assert N[0] == N[1]

def test_dtype():
    numpy.random.seed(1000)
    pos = numpy.arange(10).reshape(-1, 1)
    try:
        tree = KDTree(pos).root
        assert True
    except TypeError:
        pass

def test_ind():
    numpy.random.seed(1000)
    pos = numpy.arange(10).reshape(-1, 1).astype('f4')
    tree = KDTree(pos, ind=[0, 1, 2, 3])
    assert tree.root.size == 4

def test_count():
    numpy.random.seed(1234)
    pos1 = numpy.random.uniform(size=(1000, 3)).astype('f8')
    pos2 = numpy.random.uniform(size=(1000, 3)).astype('f8')

    dist = ((pos1[:, None, :] - pos2[None, :, :]) ** 2).sum(axis=-1)
    truth = (dist < 1).sum()

    tree1 = KDTree(pos1).root
    tree2 = KDTree(pos2).root
    c = tree1.count(tree2, r=1.0)
    assert_equal(c, truth)

def test_enum():
    numpy.random.seed(1234)
    pos1 = numpy.random.uniform(size=(1000, 3)).astype('f8')
    pos2 = numpy.random.uniform(size=(1000, 3)).astype('f8')
    dist = ((pos1[:, None, :] - pos2[None, :, :]) ** 2).sum(axis=-1)
    truth = (dist < 1).sum()

    tree1 = KDTree(pos1).root
    tree2 = KDTree(pos2).root
    N = [0]
    def process1(r, i, j):
        N[0] += (r < 1.0).sum()

    tree1.enum(tree2, rmax=1.0, process=process1)
    assert_equal(N[0], truth)

def test_enum_count_weighted():
    numpy.random.seed(1234)
    pos1 = numpy.random.uniform(size=(1000, 3)).astype('f4')
    pos2 = numpy.random.uniform(size=(1000, 3)).astype('f4')
    w1 = numpy.ones(len(pos1))
    w2 = numpy.ones(len(pos2))
    tree1 = KDTree(pos1)
    tree2 = KDTree(pos2)
    a1 = KDAttr(tree1, w1)
    a2 = KDAttr(tree2, w2)
    N = [0]
    def process1(r, i, j):
        N[0] += len(r)
    tree1.root.enum(tree2.root, rmax=1.0, process=process1)
    c, w = tree1.root.count(tree2.root, r=1.0, attrs=(a1, a2))
    assert_equal(N[0], c)
    assert_equal(N[0], w)

def test_count_symmetric():
    pos1 = numpy.random.uniform(size=(1000000, 3)).astype('f4')
    pos2 = numpy.array([[0.3, 0.5, 0.1]], dtype='f4')
    tree1 = KDTree(pos1).root
    tree2 = KDTree(pos2).root
    assert_equal(tree2.count(tree1, (0, 0.1, 1.0)),
                 tree1.count(tree2, (0, 0.1, 1.0)))

def test_integrate1d():
    pos = numpy.arange(1000).astype('f4').reshape(-1, 1)
    tree = KDTree(pos)
    root = tree.root
    assert_equal(root.integrate(-numpy.inf, numpy.inf), 1000)

    assert_equal(root.integrate(0, 1), 1)
    assert_equal(root.integrate(0, 0), 0)
    assert_equal(root.integrate(999, 999), 0)
    assert_equal(root.integrate(0, pos), pos[:, 0])

def test_integrate2d():
    N = 1000
    pos = numpy.arange(N).astype('f4').reshape(-1, 1)
    pos = numpy.concatenate([pos, pos], axis=-1)
    tree = KDTree(pos)
    root = tree.root
    assert_equal(root.integrate(-numpy.inf, numpy.inf), N)

    assert_equal(root.integrate(0, pos), numpy.arange(N))

def test_empty():
    pos = numpy.arange(1000).astype('f4').reshape(-1, 1)
    shape = ()

    data = numpy.ones((len(pos)), dtype=('f4', shape))
    tree = KDTree(pos, ind=[])
    attr = KDAttr(tree, data)
    assert_equal(attr[tree.root], 0)

def test_attr():
    pos = numpy.arange(1000).astype('f4').reshape(-1, 1)
    shapes = [(), (1,), (2,)]
    for shape in shapes:
        data = numpy.empty((len(pos)), dtype=('f4', shape))
        data[:] = 1.0

        tree = KDTree(pos)
        attr = KDAttr(tree, data)
        assert_equal(tree.root.index, 0)
        assert_equal(tree.root.less.index, 1)
        assert_equal(tree.root.greater.index, 2)
        assert_equal(attr.buffer.shape[0], tree.size)
        assert_equal(attr.buffer.shape[1:], shape)
        assert_equal(attr[tree.root], data.sum(axis=0))

def test_constattr():
    pos = numpy.arange(100).astype('f4').reshape(-1, 1)
    shapes = [(), (1,), (2,)]
    for shape in shapes:
        data = constant_array((len(pos)), dtype=('f4', shape))
        data.value[...] = 1.0

        tree = KDTree(pos)
        attr = KDAttr(tree, data)
        assert_equal(tree.root.index, 0)
        assert_equal(tree.root.less.index, 1)
        assert_equal(tree.root.greater.index, 2)
        assert_equal(attr.buffer.shape[0], tree.size)
        assert_equal(attr.buffer.shape[1:], shape)
        assert_equal(attr[tree.root], data.sum(axis=0))

def test_fof():
    numpy.random.seed(1000)
    pos = numpy.linspace(0, 1, 10000, endpoint=False).reshape(-1, 1)
    tree = KDTree(pos).root

    label = tree.fof(1.1/ len(pos))
    assert_equal(numpy.unique(label).size, 1)

    label = tree.fof(0.8/ len(pos))
    assert_equal(numpy.unique(label).size, len(pos))

def test_force():
    from kdcount import force_kernels

    pos = numpy.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    #pos = numpy.arange(0, 4, dtype='f8').reshape(-1, 1)
    tree = KDTree(pos, thresh=1)
    mass = KDAttr(tree, numpy.ones(pos.shape[0]))
    xmass = KDAttr(tree, pos * mass.input)

    force = tree.root.force(force_kernels['plummer'](1), pos, mass, xmass, 2., eta=0.1)
    # FIXME: add more tests
    print(force)

def test_force_slow():
    from kdcount import force_kernels

    numpy.random.seed(13)
    pos = numpy.random.uniform(size=(32 * 32 * 32, 3)) 
    tree = KDTree(pos, thresh=1)
    mass = KDAttr(tree, numpy.ones(pos.shape[0]))
    xmass = KDAttr(tree, pos * mass.input)

    #force = tree.root.force(force_kernels['plummer'](1), pos, mass, xmass, 1.0 / len(pos) ** 0.3333 * 4, eta=0.1)
    force = tree.root.force(force_kernels['count'], pos, mass, xmass, 1.0 / len(pos) ** 0.3333 * 4, eta=0.1)

    # FIXME: add more tests
    print(force)

def test_force2_slow():
    from kdcount import force_kernels
    from time import time
    numpy.random.seed(13)
    pos = numpy.random.uniform(size=(32 * 32 * 32 * 8, 3)) 
    tt = KDTree(pos[:1], thresh=1)
    tree = KDTree(pos, thresh=1)
    mass = KDAttr(tree, numpy.ones(pos.shape[0]))
    xmass = KDAttr(tree, pos * mass.input)
    print(tt.root, tt.root.min, tt.root.max)

    t0 = time()
    force2 = tree.root.force2(force_kernels['count'], tt.root, mass, xmass, 1.0 / len(pos) ** 0.3333 * 8, eta=1.)
    t2 = time() - t0
    print('time', t2)
    t0 = time()
    force1 = tree.root.force(force_kernels['count'], pos[:1], mass, xmass, 1.0 / len(pos) ** 0.3333 * 8, eta=1.)
    t1 = time() - t0
    print('time', t1)
    # FIXME: add more tests
    assert_array_equal(force1, force2)


if __name__ == "__main__":
    run_module_suite()
