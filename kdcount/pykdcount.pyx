#cython: embedsignature=True
#cython: cdivision=True
cimport numpy
import numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from numpy cimport npy_intp
from numpy cimport npy_uint64
numpy.import_array()

cdef extern from "kdtree.h":
    struct KDEnumData:
        double r 
        npy_intp i
        npy_intp j

    ctypedef double (*kd_castfunc)(void * p)
    ctypedef int (*kd_enum_callback)(void * data, KDEnumData * endata)
    ctypedef void (*kd_freefunc)(void* data, npy_intp size, void * ptr) nogil
    ctypedef void * (*kd_mallocfunc)(void* data, npy_intp size) nogil

    struct cKDArray "KDArray":
        char * buffer
        npy_intp dims[2]
        npy_intp strides[2]
        npy_intp elsize
        double (* cast)(void * p1)
    
    struct cKDTree "KDTree":
        cKDArray input
        int thresh
        npy_intp * ind
        double * boxsize
        double p
        kd_mallocfunc malloc
        kd_freefunc  free
        void * userdata
        npy_intp size 

    struct cKDAttr "KDAttr":
        cKDTree * tree
        cKDArray input
        double * buffer

    struct cKDNode "KDNode":
        cKDTree * tree
        cKDNode * link[2]
        npy_intp index
        npy_intp start
        npy_intp size
        int dim
        double split

    cKDNode * kd_build(cKDTree * tree) nogil
    double * kd_node_max(cKDNode * node) nogil
    double * kd_node_min(cKDNode * node) nogil
    void kd_free(cKDNode * node) nogil
    void kd_free0(cKDTree * tree, npy_intp size, void * ptr) nogil

    int kd_enum(cKDNode * nodes[2], double maxr,
            kd_enum_callback callback, void * data) except -1

    int kd_fof(cKDNode * tree, double linklength, npy_intp * head) nogil

    void kd_attr_init(cKDAttr * attr, cKDNode * root) nogil
    void kd_count(cKDNode * nodes[2], 
            cKDAttr * attrs[2], 
            double * edges, npy_uint64 * count, 
            double * weight, 
            int nedges) nogil

cdef class KDNode:
    cdef cKDNode * ref
    cdef readonly KDTree tree 
    cdef readonly npy_intp ndims

    def __init__(self, KDTree tree):
        self.tree = tree
        self.ndims = tree.ndims    

    cdef void bind(self, cKDNode * ref) nogil:
        self.ref = ref

    property index:
        def __get__(self):
            return self.ref.index

    property less:
        def __get__(self):
            cdef KDNode rt = type(self)(self.tree)
            if self.ref.link[0]:
                rt.bind(self.ref.link[0])
                return rt
            else:
                return None

    property greater:
        def __get__(self):
            cdef KDNode rt = type(self)(self.tree)
            if self.ref.link[1]:
                rt.bind(self.ref.link[1])
                return rt
            else:
                return None

    property start:
        def __get__(self):
            return self.ref.start
    
    property size:
        def __get__(self):
            return self.ref.size

    property dim:
        def __get__(self):
            return self.ref.dim

    property split:
        def __get__(self):
            return self.ref.split

    property max:
        def __get__(self):
            cdef double * max = kd_node_max(self.ref)
            return numpy.array([max[d] for d in
                range(self.ref.tree.input.dims[1])])

    property min:
        def __get__(self):
            cdef double * min = kd_node_min(self.ref)
            return numpy.array([min[d] for d in
                range(self.ref.tree.input.dims[1])])

    def __richcmp__(self, other, int op):
        if op == 0: return False
        if op == 1: return True
        if op == 2: return True
        if op == 3: return False
        if op == 4: return False
        if op == 5: return True

    def __repr__(self):
        return str(('%X' % <npy_intp>self.ref, self.dim, self.split, self.size))

    def count(self, KDNode other, r, attrs=None):
        cdef:
            numpy.ndarray r1, count, weight
            KDAttr a1, a2
            cKDNode * cnodes[2]
            cKDAttr * cattrs[2]

        r = numpy.array(r)

        r1 = numpy.empty(r.shape, dtype='f8')
        count = numpy.zeros(r.shape, dtype='u8')
        cnodes[0] = self.ref
        cnodes[1] = other.ref

        r1[...] = r
        if attrs is None:
            cattrs[0] = NULL
            cattrs[1] = NULL
            kd_count(cnodes, cattrs, 
                    <double*>r1.data, 
                    <npy_uint64*>count.data,
                    NULL, 
                    r1.size)

            return count
        else:
            if isinstance(attrs, (tuple, list)):
                a1, a2 = attrs
            else:
                if self.tree != other.tree:
                    raise ValueError("Must be the same tree if one weight is used")
                a1 = a2 = attrs

            if a1.ndims != a2.ndims:
                raise ValueError("Two attributes must have the same dimensions")

            cattrs[0] = a1.ref
            cattrs[1] = a2.ref
            weight = numpy.zeros(r.shape, dtype=('f8', a1.ndims))

            kd_count(cnodes, cattrs, 
                    <double*>r1.data, 
                    <npy_uint64*>count.data,
                    <double*>weight.data, 
                    r1.size)

            return count, weight

    def fof(self, double linkinglength, out=None):
        cdef numpy.ndarray buf
        if out is not None:
            assert out.dtype == numpy.dtype('intp')
            buf = out
        else:
            buf = numpy.empty(self.size, dtype='intp')
        if -1 == kd_fof(self.ref, linkinglength, <npy_intp*> buf.data):
            raise RuntimeError("Too many friend of friend iterations. This is likely a bug.");
        return buf


    def enum(self, KDNode other, rmax, process=None, bunch=100000, **kwargs):
        """ cross correlate with other, for all pairs
            closer than rmax, iterate.

            >>> def process(r, i, j, **kwargs):
            >>>    ...

            >>> A.enum(... process, **kwargs):
            >>>   ...

            where r is the distance, i and j are the original
            input array index of the data. arbitrary args can be passed
            to process via kwargs.
        """
        cdef:
             numpy.ndarray r, i, j
             cKDNode * node[2]
             CBData cbdata

        r = numpy.empty(bunch, 'f8')
        i = numpy.empty(bunch, 'intp')
        j = numpy.empty(bunch, 'intp')

        rall = None
        if process is None:
            rall = [numpy.empty(0, 'f8')]
            iall = [numpy.empty(0, 'intp')]
            jall = [numpy.empty(0, 'intp')]
            def process(r1, i1, j1, **kwargs):
                rall[0] = numpy.append(rall[0], r1)
                iall[0] = numpy.append(iall[0], i1)
                jall[0] = numpy.append(jall[0], j1)

        def func():
            process(r[:cbdata.length].copy(), 
                    i[:cbdata.length].copy(), 
                    j[:cbdata.length].copy(),
                    **kwargs)
        node[0] = self.ref
        node[1] = other.ref
        cbdata.notify = <void*>func
        cbdata.ndims = self.ndims
        cbdata.r = <double*>r.data
        cbdata.i = <npy_intp*>i.data
        cbdata.j = <npy_intp*>j.data
        cbdata.size = bunch
        cbdata.length = 0
        kd_enum(node, rmax, <kd_enum_callback>callback, &cbdata)

        if cbdata.length > 0:
            func()

        if rall is not None:
            return rall[0], iall[0], jall[0]
        else:
            return None

cdef double dcast(double * p1) nogil:
    return p1[0]
cdef double fcast(float * p1) nogil:
    return p1[0]

cdef struct CBData:
    double * r
    npy_intp * i
    npy_intp * j
    double * x
    double * y
    npy_intp size
    npy_intp length
    void * notify
    int ndims

cdef int callback(CBData * data, KDEnumData * endata) except -1:
    if data.length == data.size:
        (<object>(data.notify)).__call__()
        data.length = 0
    cdef int d
    data.r[data.length] = endata.r
    data.i[data.length] = endata.i
    data.j[data.length] = endata.j

    data.length = data.length + 1
    return 0

cdef class KDAttr:
    cdef cKDAttr * ref
    cdef readonly numpy.ndarray input
    cdef readonly KDTree tree
    cdef readonly npy_intp ndims
    cdef readonly numpy.ndarray buffer

    def __init__(self, KDTree tree, numpy.ndarray input):
        self.tree = tree
        if input.ndim == 1:
            input = input.reshape(-1, 1) 
            self.buffer = numpy.empty(tree.ref.size, dtype='f8')
        elif input.ndim == 2:
            self.buffer = numpy.empty(tree.ref.size, dtype=('f8', (<object>input).shape[1:]))
        else:
            raise ValueError("Only at most 2d attribute array is supported")

        self.ndims = input.shape[1]

        self.input = input
        self.ref = <cKDAttr*>PyMem_Malloc(sizeof(cKDAttr))
        self.ref.tree = tree.ref
        self.ref.input.buffer = input.data
        self.ref.input.dims[0] = input.shape[0]
        self.ref.input.dims[1] = input.shape[1]
        self.ref.input.strides[0] = input.strides[0]
        self.ref.input.strides[1] = input.strides[1]
        self.ref.input.elsize = input.dtype.itemsize
        if input.dtype.char == 'f':
            self.ref.input.cast = <kd_castfunc>fcast
        if input.dtype.char == 'd':
            self.ref.input.cast = <kd_castfunc>dcast

        self.ref.buffer = <double *> self.buffer.data 

        kd_attr_init(self.ref, tree._root);

    def __getitem__(self, node):
        cdef KDNode _node
        if isinstance(node, KDNode):
            _node = node
            assert _node.ref.tree == self.tree.ref
            return self.buffer[_node.ref.index]
        else:
            raise KeyError("Only node can be used.")
    def __dealloc__(self): 
        PyMem_Free(self.ref)
        
cdef class KDTree:
    cdef cKDTree * ref
    cdef cKDNode * _root 
    cdef readonly numpy.ndarray input
    cdef readonly numpy.ndarray ind
    cdef readonly numpy.ndarray boxsize
    cdef readonly npy_intp ndims
 
    __nodeclass__ = KDNode

    property root:
        def __get__(self):
            cdef KDNode rt = type(self).__nodeclass__(self)
            rt.bind(self._root)
            return rt
    property size:
        def __get__(self):
            return self.ref.size

    # memory management:

    cdef readonly list buffers
    cdef readonly char * _bufferptr
    cdef readonly npy_intp  _free

    cdef void _addbuffer(self):
        cdef numpy.ndarray buffer = numpy.empty(1024 * 1024, dtype='u1')
        self.buffers.append(buffer)
        self._bufferptr = <char*> (buffer.data)
        self._free = 1024 * 1024

    cdef void * malloc(self, npy_intp size) nogil:
        cdef void * ptr
        if size > self._free:
            with gil:
                self._addbuffer()
        self._free -= size
        ptr = <void*> self._bufferptr
        self._bufferptr += size
        return ptr

    cdef void free(self, npy_intp size, void * ptr) nogil:
        # do nothing
        return

    def __cinit__ (self):
        self.buffers = []
    
    def __init__(self, numpy.ndarray input, boxsize=None, thresh=10):
        if input.ndim != 2:
            raise ValueError("input needs to be a 2 D array of (N, Nd)")
        self.input = input
        self.thresh = thresh
        self.ref = <cKDTree*>PyMem_Malloc(sizeof(cKDTree))
        self.ref.input.buffer = input.data
        self.ref.input.dims[0] = input.shape[0]
        self.ref.input.dims[1] = input.shape[1]
        self.ref.input.strides[0] = input.strides[0]
        self.ref.input.strides[1] = input.strides[1]
        self.ref.input.elsize = input.dtype.itemsize
        if input.dtype.char == 'f':
            self.ref.input.cast = <kd_castfunc>fcast
        if input.dtype.char == 'd':
            self.ref.input.cast = <kd_castfunc>dcast

        self.ndims = self.input.shape[1]

        self.ref.thresh = thresh
        self.ind = numpy.empty(self.ref.input.dims[0], dtype='intp')
        self.ref.ind = <npy_intp*> self.ind.data
        if boxsize is not None:
            self.boxsize = numpy.empty(self.ref.input.dims[1], dtype='f8')
            self.boxsize[:] = boxsize
            self.ref.boxsize = <double*>self.boxsize.data
        else:
            self.boxsize = None
            self.ref.boxsize = NULL
        self.ref.userdata = <void*> self
        self.ref.malloc = <kd_mallocfunc> self.malloc
        self.ref.free = <kd_freefunc> self.free
        with nogil:
            self._root = kd_build(self.ref)

    def __dealloc__(self):
        # self.buffers will be freed by cython
        # no need to call kd_free !
        PyMem_Free(self.ref)

