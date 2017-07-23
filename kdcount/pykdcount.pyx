#cython: embedsignature=True
#cython: cdivision=True
cimport numpy
import numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
cimport cython
from numpy cimport npy_intp
from numpy cimport npy_uint64
numpy.import_array()

cdef extern from "kdtree.h":
    struct KDEnumPair:
        double r 
        npy_intp i
        npy_intp j

    struct KDEnumNodePair:
        pass

    ctypedef double (*kd_castfunc)(void * p)
    ctypedef int (*kd_enum_visit_edge)(void * userdata, KDEnumPair * pair)
    ctypedef int (*kd_enum_prune_nodes)(void * userdata, KDEnumNodePair * pair, int * open)
    ctypedef void (*kd_freefunc)(void* data, npy_intp size, void * ptr) nogil
    ctypedef void * (*kd_mallocfunc)(void* data, npy_intp size) nogil

    ctypedef int (*kd_point_point_cullmetric)(void * userdata, int ndims, double * dx, double * dist) nogil
    ctypedef int (*kd_node_node_cullmetric)(void * userdata, int ndims, double * min, double * max,
            double * distmin, double * distmax) nogil

    ctypedef void (*kd_force_func)(double r, double * dx, double * f, int ndims, void * userdata)

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
        npy_intp ind_size
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
            kd_enum_visit_edge visit_edge,
            kd_enum_prune_nodes prune_nodes,
            void * data) except -1

    int kd_fof(cKDNode * tree, double linklength, npy_intp * head) nogil
    int kd_fof_unsafe(cKDNode * tree, double linklength, npy_intp * head) nogil
    int kd_fof_allpairs(cKDNode * tree, double linklength, npy_intp * head) nogil
    int kd_fof_prefernodes(cKDNode * tree, double linklength, npy_intp * head) nogil
    int kd_fof_heuristics(cKDNode * tree, double linklength, npy_intp * head) nogil
    int kd_fof_buggy(cKDNode * tree, double linklength, npy_intp * head) nogil
    int kd_fof_linkedlist(cKDNode * tree, double linklength, npy_intp * head) nogil
    void kd_fof_get_last_traverse_info(npy_intp *visited, npy_intp *enumerated, npy_intp *connected,
                                  npy_intp *maxdepth, npy_intp *nsplay, npy_intp *totaldepth)

    void kd_attr_init(cKDAttr * attr, cKDNode * root) nogil
    void kd_count(cKDNode * nodes[2], 
            cKDAttr * attrs[2], 
            double * edges, npy_uint64 * count, 
            double * weight, 
            int nedges, 
            kd_point_point_cullmetric ppcull, 
            kd_node_node_cullmetric nncull, void * userdata,
            npy_uint64 * brute_force,
            npy_uint64 * node_node) nogil
    void kd_integrate(cKDNode * node, 
            cKDAttr * attr, 
            npy_uint64 * count, 
            double * weight, 
            double * min, double * max,
            npy_uint64 * brute_force,
            npy_uint64 * node_node) nogil

    void kd_forcea(double * pos, ptrdiff_t n, cKDNode * node, cKDAttr * mass, cKDAttr * xmass,
            double r_cut, double eta, double * force,
            kd_force_func func, void * userdata)

    void kd_force2(cKDNode * target, cKDNode * node, cKDAttr * mass, cKDAttr * xmass,
            double r_cut, double eta, double * force,
            kd_force_func func, void * userdata)

cdef class KDForceKernel:
    cdef kd_force_func func
    cdef double parameters[10]

cdef void kd_force_func_count(double r, double * dx, double * f, int ndims, void * userdata):
    for i in range(ndims):
        f[i] = 1

cdef class KDForceCount(KDForceKernel):
    def __init__(self):
        self.func = kd_force_func_count

cdef void kd_force_func_plummer(double r, double * dx, double * f, int ndims, void * userdata):
    cdef double ir3
    cdef double * parameters = <double*> userdata
    cdef double a = parameters[0]

    if r > 0:
        ir3 = 1.0 / (r**2 + a**2)**1.5
    else :
        ir3 = 0

    for i in range(ndims):
        f[i] = dx[i] * ir3

cdef class KDForcePlummer(KDForceKernel):
    def __init__(self, a):
        self.func = kd_force_func_plummer
        self.parameters[0] = a

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

    def count(self, KDNode other, numpy.ndarray r, attrs, info={}):
        cdef:
            numpy.ndarray count, weight
            KDAttr a1, a2
            cKDNode * cnodes[2]
            cKDAttr * cattrs[2]
            npy_uint64 brute_force
            npy_uint64 node_node
        assert r.dtype == numpy.dtype('f8')
        count = numpy.zeros((<object>r).shape, dtype='u8')
        cnodes[0] = self.ref
        cnodes[1] = other.ref

        if attrs is None:
            cattrs[0] = NULL
            cattrs[1] = NULL
            kd_count(cnodes, cattrs, 
                    <double*>r.data, 
                    <npy_uint64*>count.data,
                    NULL, 
                    r.size, 
                    <kd_point_point_cullmetric>NULL,
                    <kd_node_node_cullmetric>NULL,
                    NULL, &brute_force, &node_node)
            info['brute_force'] = brute_force
            info['node_node'] = node_node
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
            dtype = numpy.dtype(('f8', a1.ndims))
            weight = numpy.zeros((<object>r).shape, dtype=dtype)

            kd_count(cnodes, cattrs, 
                    <double*>r.data, 
                    <npy_uint64*>count.data,
                    <double*>weight.data, 
                    r.size, 
                    <kd_point_point_cullmetric>NULL,
                    <kd_node_node_cullmetric>NULL,
                    NULL, &brute_force, &node_node)

            info['brute_force'] = brute_force
            info['node_node'] = node_node

            return count, weight

    def fof(self, double linkinglength, numpy.ndarray out, method):
        assert out.dtype == numpy.dtype('intp')

        if method == 'linkedlist':
            rt = kd_fof_linkedlist(self.ref, linkinglength, <npy_intp*> out.data)
        elif method == 'unsafe':
            rt = kd_fof_unsafe(self.ref, linkinglength, <npy_intp*> out.data)
        elif method == 'allpairs':
            rt = kd_fof_allpairs(self.ref, linkinglength, <npy_intp*> out.data)
        elif method == 'heuristics':
            rt = kd_fof_heuristics(self.ref, linkinglength, <npy_intp*> out.data)
        elif method == 'buggy':
            rt = kd_fof_buggy(self.ref, linkinglength, <npy_intp*> out.data)
        elif method == 'prefernodes':
            rt = kd_fof_prefernodes(self.ref, linkinglength, <npy_intp*> out.data)
        else:
            rt = kd_fof(self.ref, linkinglength, <npy_intp*> out.data)

        if rt == -1:
            raise RuntimeError("Friend of Friend failed. This is likely a bug.")

        return out

    def integrate(self, numpy.ndarray min, numpy.ndarray max, KDAttr attr, info={}):
        cdef:
            numpy.ndarray count, weight
            cKDAttr * cattr
            npy_intp i
            npy_uint64 brute_force, _brute_force
            npy_uint64 node_node, _node_node
            npy_intp N
        shape = (<object>min).shape[:-1]
        count = numpy.zeros(shape, dtype='u8')
        if min.ndim == 1:
            N = 1
        else:
            N = len(min)

        brute_force = 0
        node_node = 0
        info['brute_force'] = 0
        info['node_node'] = 0

        if attr is None:
            cattr = NULL
            for i in range(N):
                kd_integrate(self.ref, cattr,
                    (<npy_uint64*> count.data) + i,
                    NULL,
                    <double*> ((<char*> min.data) + i * min.strides[0]),
                    <double*> ((<char*> max.data) + i * max.strides[0]),
                    &_brute_force, &_node_node)
                brute_force += _brute_force
                node_node += _node_node

            result = count
        else:
            cattr = attr.ref
            dtype = numpy.dtype(('f8', attr.ndims))
            weight = numpy.zeros(shape, dtype=dtype)

            for i in range(N):
                kd_integrate(self.ref, cattr,
                        (<npy_uint64*> count.data) + i,
                        (<double*> weight.data) + i * attr.ndims,
                        <double*> ((<char*> min.data) + i * min.strides[0]),
                        <double*> ((<char*> max.data) + i * max.strides[0]),
                        &_brute_force, &_node_node)
                brute_force += _brute_force
                node_node += _node_node

            result = count, weight

        info['brute_force'] = brute_force
        info['node_node'] = node_node

        return result

    def force(self, KDForceKernel kernel, cython.floating [:, ::1] pos, KDAttr mass, KDAttr xmass,
                double r_cut, double eta=0.2):
        cdef npy_intp N, i, d
        cdef numpy.ndarray force
        cdef double x[32]

        force = numpy.zeros((pos.shape[0], pos.shape[1]), dtype='f8')

        for i in range(pos.shape[0]):
            for d in range(pos.shape[1]):
                x[d] = pos[i, d]

                kd_forcea(&x[0], 1, self.ref, mass.ref, xmass.ref,
                    r_cut, eta,
                    <double*> ((<char*> force.data) + force.strides[0] * i),
                    kernel.func, <void*> kernel.parameters)

        return force

    def force2(self, KDForceKernel kernel, KDNode target, KDAttr mass, KDAttr xmass,
                double r_cut, double eta=0.2, out=None):
        if out is None:
            out = numpy.zeros((target.tree.input.shape[0], target.tree.input.shape[1]), dtype='f8')

        assert out.dtype == numpy.dtype('f8')

        cdef numpy.ndarray force = out

        kd_force2(target.ref, self.ref, mass.ref, xmass.ref,
            r_cut, eta,
            <double*> force.data,
            kernel.func, <void*> kernel.parameters)

        return force

    def enum(self, KDNode other, rmax, process, bunch, **kwargs):
        cdef:
             numpy.ndarray r, i, j
             cKDNode * node[2]
             CBData cbdata

        r = numpy.empty(bunch, 'f8')
        i = numpy.empty(bunch, 'intp')
        j = numpy.empty(bunch, 'intp')

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
        kd_enum(node, rmax, <kd_enum_visit_edge>callback, NULL, &cbdata)

        if cbdata.length > 0:
            func()


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

cdef int callback(CBData * data, KDEnumPair * pair) except -1:
    if data.length == data.size:
        (<object>(data.notify)).__call__()
        data.length = 0
    cdef int d
    data.r[data.length] = pair.r
    data.i[data.length] = pair.i
    data.j[data.length] = pair.j

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
            dtype = numpy.dtype(('f8', (<object>input).shape[1:]))
            self.buffer = numpy.empty(tree.ref.size, dtype=dtype)
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
            self.ref.input.cast = NULL #<kd_castfunc>dcast

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
            if self._root == NULL:
                return None

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
        self.ref = <cKDTree*>PyMem_Malloc(sizeof(cKDTree))
        self._root = NULL

    def __init__(self, numpy.ndarray input, boxsize=None, thresh=10, ind=None):
        if input.ndim != 2:
            raise ValueError("input needs to be a 2 D array of (N, Nd)")
        self.input = input
        self.thresh = thresh
        self.ref.input.buffer = input.data
        self.ref.input.dims[0] = input.shape[0]
        self.ref.input.dims[1] = input.shape[1]
        self.ref.input.strides[0] = input.strides[0]
        self.ref.input.strides[1] = input.strides[1]
        self.ref.input.elsize = input.dtype.itemsize
        if input.dtype.char == 'f':
            self.ref.input.cast = <kd_castfunc>fcast
        elif input.dtype.char == 'd':
            self.ref.input.cast = NULL # <kd_castfunc>dcast
        else:
            raise TypeError("input type of %s is unsupported" % input.dtype)

        self.ndims = self.input.shape[1]

        self.ref.thresh = thresh
        
        if ind is None:
            ind = numpy.arange(self.ref.input.dims[0], dtype='intp')
        else:
            # always make a copy to avoid overwritting the original
            ind = numpy.array(ind, dtype='intp', copy=True)

        self.ind = ind
        self.ref.ind_size = len(self.ind)
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

def _get_last_fof_info():
    cdef:
         npy_intp visited
         npy_intp connected 
         npy_intp enumerated 
         npy_intp maxdepth
         npy_intp nsplay
         npy_intp totaldepth

    kd_fof_get_last_traverse_info(&visited, &enumerated, &connected, &maxdepth, &nsplay, &totaldepth)
    return (visited, enumerated, connected, maxdepth, nsplay, totaldepth)
