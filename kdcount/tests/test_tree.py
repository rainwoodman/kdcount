from kdcount import KDTree, KDAttr
import numpy
from numpy.testing import assert_equal, run_module_suite

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

def test_count_symmetric():
    pos1 = numpy.random.uniform(size=(1000000, 3)).astype('f4')
    pos2 = numpy.array([[0.3, 0.5, 0.1]], dtype='f4')
    tree1 = KDTree(pos1).root
    tree2 = KDTree(pos2).root
    assert_equal(tree2.count(tree1, (0, 0.1, 1.0)),
                 tree1.count(tree2, (0, 0.1, 1.0)))

def test_attr():
    pos = numpy.arange(100).astype('f4').reshape(-1, 1)
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

if __name__ == "__main__":
    run_module_suite()
