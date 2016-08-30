from kdcount import KDTree, KDAttr
import numpy
from numpy.testing import assert_equal, run_module_suite
from kdcount.utils import constant_array

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
def test_enum_count_agree():
    pos1 = numpy.random.uniform(size=(1000, 3)).astype('f4')
    pos2 = numpy.random.uniform(size=(1000, 3)).astype('f4')
    tree1 = KDTree(pos1).root
    tree2 = KDTree(pos2).root
    N = [0]
    def process1(r, i, j):
        N[0] += len(r)
    tree1.enum(tree2, rmax=1.0, process=process1)
    c = tree1.count(tree2, r=1.0)
    assert_equal(N[0], c)

def test_enum_count_weighted():
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

if __name__ == "__main__":
    run_module_suite()
