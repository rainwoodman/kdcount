#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

typedef double (*kd_castfunc)(void * p);
/* x and y points to the double position */
typedef struct KDEnumData {
    double r;
    ptrdiff_t i;
    ptrdiff_t j;
} KDEnumData;

typedef int (*kd_enum_callback)(void * data, KDEnumData * enumdata);


typedef struct KDStore {

/* defining the input positions */

    /* the buffer holding point positions  required */
    char * buffer; 
    /* number of points. required*/
    ptrdiff_t size;
    /* number of dimensions of each point. required */
    int Nd;
    /* the byte offset of the axes.  required
     * the i-th position , d-th component is
     * at i * strides[0] + d * strides[1] */
    ptrdiff_t strides[2]; 

/* the following defines how the tree is constructed */

    /* split thresh, required. 10 is good*/
    int thresh;
    /* a permutation array for sorting, required*/
    ptrdiff_t * ind; 
    /* the periodic boxsize per axis,
     * or NULL if there is no box  */
    double * boxsize;
    /* unused */
    double p;

/* the following defines the datatype of a position scalar */

    /* the byte size of each position scalar, required */
    ptrdiff_t elsize;
    /* if cast p1 to double and return it */
    double (* cast)(void * p1);

/* memory allocation */
    /* allocate memory, NULL to use malloc() */
    void * (* malloc)(size_t size);
    /* deallocate memory, size is passed in for a slab allocator,
     * NULL to use free() */
    void (* free)(size_t size, void * ptr);
    void * userdata;
    ptrdiff_t total_nodes;
} KDStore;

typedef struct KDNode {
    KDStore * store;
    struct KDNode * link[2];
    ptrdiff_t start;
    ptrdiff_t size;
    int dim;
    double split;
    char ext[];
} KDNode;

static void * kd_malloc(KDStore * store, size_t size) {
    if(store->malloc != NULL) {
        return store->malloc(size);
    } else {
        return malloc(size);
    }
}

static KDNode * kd_alloc(KDStore * store) {
    KDNode * ptr = kd_malloc(store, sizeof(KDStore) + sizeof(double) * 2 * store->Nd);
    ptr->link[0] = NULL;
    ptr->link[1] = NULL;
    ptr->store = store;
    store->total_nodes ++;
    return ptr;
}

static inline double * kd_node_max(KDNode * node) {
    return (double*) (node->ext);
}
static inline double * kd_node_min(KDNode * node) {
    return kd_node_max(node) + node->store->Nd;
}

static inline void * kd_ptr(KDStore * store, ptrdiff_t i, ptrdiff_t d) {
    i = store->ind[i];
    return & store->buffer[i * store->strides[0] + d * store->strides[1]];
}
static inline double kd_cast(KDStore * store, void * p1) {
    return store->cast(p1);
}
static inline double kd_data(KDStore * store, ptrdiff_t i, ptrdiff_t d) {
    i = store->ind[i];
    char * ptr = & store->buffer[i * store->strides[0] + d * store->strides[1]];
    return kd_cast(store, ptr);
}
static inline void kd_swap(KDStore * store, ptrdiff_t i, ptrdiff_t j) {
    ptrdiff_t t = store->ind[i];
    store->ind[i] = store->ind[j];
    store->ind[j] = t;
}

static void kd_build_split(KDNode * node, double minhint[], double maxhint[]) {
    KDStore * store = node->store;
    ptrdiff_t p, q, j;
    int d;

    double * max = kd_node_max(node);
    double * min = kd_node_min(node);
    for(d = 0; d < store->Nd; d++) {
        max[d] = maxhint[d];
        min[d] = minhint[d];
    }

    if(node->size <= store->thresh) {
        node->dim = -1;
        return;
    }

    node->dim = 0;
    double longest = maxhint[0] - minhint[0];
    for(d = 1; d < store->Nd; d++) {
        double tmp = maxhint[d] - minhint[d];
        if(tmp > longest) {
            node->dim = d;
            longest = tmp;
        }
    }

    node->split = (max[node->dim] + min[node->dim]) * 0.5;

    /*
    printf("trysplit @ %g (%g %g %g %g %g %g) dim = %d, %td %td\n",
            node->split, 
            max[0], 
            max[1], 
            max[2], 
            min[0],  
            min[1],  
            min[2],  
            node->dim, node->start, node->size);
    */
    p = node->start;
    q = node->start + node->size - 1;
    while(p <= q) {
        if(kd_data(store, p, node->dim) < node->split) {
            p ++;
        } else if(kd_data(store, q, node->dim) >= node->split) {
            q --;
        } else {
            kd_swap(store, p, q); 
            p ++;
            q --;
        }
    }
    /* invariance: data[<p] < split and data[>q] >= split
     * after loop p > q.
     * thus data[0...,  p-1] < split
     * data[q + 1...., end] >= split
     * p - 1 < q + 1
     * p < q + 2
     * p > q
     * thus p = q + 1 after the loop.
     *
     * 0 -> p -1 goes to left
     * and p -> end goes to right, so that
     *  left < split
     *  and right >= split
     *  */

    /* the invariance is broken after sliding.
     * after sliding we have data[0, .... p - 1] <= split
     * and data[q +1.... end] >= split */
    if(p == node->start) {
        q = node->start;
        for(j = node->start + 1; j < node->start + node->size; j++) {
            if (kd_data(store, j, node->dim) <
                kd_data(store, node->start, node->dim)) {
                kd_swap(store, j, node->start);
            }
        }
        node->split = kd_data(store, node->start, node->dim);
        p = q + 1;
    }
    if(p == node->start + node->size) {
        p = node->start + node->size - 1;
        for(j = node->start; j < node->start + node->size- 1; j++) {
            if (kd_data(store, j, node->dim) > 
                kd_data(store, node->start + node->size - 1, node->dim)) {
                kd_swap(store, j, node->start + node->size - 1);
            }
        }
        node->split = kd_data(store, node->start + node->size - 1, node->dim);
        q = p - 1;
    }

    node->link[0] = kd_alloc(store);
    node->link[0]->start = node->start;
    node->link[0]->size = p - node->start;
    node->link[1] = kd_alloc(store);
    node->link[1]->start = p;
    node->link[1]->size = node->size - (p - node->start);
/*
    printf("will split %g (%td %td), (%td %td)\n", 
            *(double*)split, 
            node->link[0]->start, node->link[0]->size,
            node->link[1]->start, node->link[1]->size);
*/
    double midhint[store->Nd];
    for(d = 0; d < store->Nd; d++) {
        midhint[d] = maxhint[d];
    }
    midhint[node->dim] = node->split;
    kd_build_split(node->link[0], minhint, midhint);
    for(d = 0; d < store->Nd; d++) {
        midhint[d] = minhint[d];
    }
    midhint[node->dim] = node->split;
    kd_build_split(node->link[1], midhint, maxhint);
}

/* 
 * create a KD tree based on input data specified in KDStore 
 * free it with kd_free
 * */
KDNode * kd_build(KDStore * store) {
    ptrdiff_t i;
    double min[store->Nd];
    double max[store->Nd];    
    int d;
    store->total_nodes = 0;
    store->ind[0] = 0;
    for(d = 0; d < store->Nd; d++) {
        min[d] = kd_data(store, 0, d);
        max[d] = kd_data(store, 0, d);
    }
    for(i = 0; i < store->size; i++) {
        store->ind[i] = i;
        for(d = 0; d < store->Nd; d++) {
            double data = kd_data(store, i, d);
            if(min[d] > data) { min[d] = data; }
            if(max[d] < data) { max[d] = data; }
        }
    }
    KDNode * tree = kd_alloc(store);
    tree->start = 0;
    tree->size = store->size;
    kd_build_split(tree, min, max);
    return tree;
}
/**
 * free a tree
 * this is recursive
 * */
void kd_free0(KDStore * store, size_t size, void * ptr) {
    if(store->free == NULL) {
        free(ptr);
    } else {
        store->free(size, ptr);
    }
}
void kd_free(KDNode * node) {
    if(node->link[0]) kd_free(node->link[0]);
    if(node->link[1]) kd_free(node->link[1]);
    node->store->total_nodes --;
    kd_free0(node->store, sizeof(KDNode) + node->store->elsize, node);
}

static void kd_realdiff(KDStore * store, double min, double max, double * realmin, double * realmax, int d) {
    if(store->boxsize) {
        double full = store->boxsize[d];
        double half = full * 0.5;
        /* periodic */
        /* /\/\ */
        if(max <= 0 || min >= 0) {
            /* do not pass through 0 */
            min = fabs(min);
            max = fabs(max);
            if(min > max) {
                double t = min;
                min = max;
                max = t;
            }
            if(max < half) {
                /* all below half*/
                *realmin = min;
                *realmax = max;
            } else if(min > half) {
                /* all above half */
                *realmax = full - min;
                *realmin = full - max;
            } else {
                /* min below, max above */
                *realmax = half;
                *realmin = fmin(min, full - max);
            }
        } else {
            /* pass though 0 */
            min = -min;
            if(min > max) max = min;
            if(max > half) max = half;
            *realmax = max;
            *realmin = 0;
        }
    } else {
        /* simple */
        /* \/     */
        if(max <= 0 || min >= 0) {
            /* do not pass though 0 */
            min = fabs(min);
            max = fabs(max);
            if(min < max) {
                *realmin = min;
                *realmax = max;
            } else {
                *realmin = max;
                *realmax = min;
            }
        } else {
            min = fabs(min);
            max = fabs(max);
            *realmax = fmax(max, min);
            *realmin = 0;
        }
    }

}
static inline void kd_collect(KDNode * node, double * ptr) {
    /* collect all positions into a double array, 
     * so that they can be paired quickly (cache locality!)*/
    KDStore * t = node->store;
    int d;
    ptrdiff_t j;

    char * base = t->buffer;
    for (j = 0; j < node->size; j++) {
        char * item = base + t->ind[j + node->start] * t->strides[0];
        for(d = 0; d < t->Nd; d++) {
            *ptr = kd_cast(t, item);
            ptr++;
            item += t->strides[1];
        }
    }
}

static int kd_count_force(KDNode * node[2], double * r2,
        uint64_t * count, size_t Nbins);

static int bisect_left(double key, double * r2, int N) {
    int left = 0, right = N;
    if(key < r2[0]) return 0;
    if(key > r2[N-1]) return N;
    while(right > left) {
        int mid = left + ((right - left) >> 1);
        if(key > r2[mid]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}
static int bisect_right(double key, double * r2, int N) {
    if(key < r2[0]) return 0;
    if(key > r2[N-1]) return N;
    int left = 0, right = N;
    while(right > left) {
        int mid = left + ((right - left) >> 1);
        if(key >= r2[mid]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}
int kd_count(KDNode * node[2], double * r2, 
        uint64_t * count, size_t Nbins) {
    int Nd = node[0]->store->Nd;
    double distmax = 0, distmin = 0;
    int d;
    double *min0 = kd_node_min(node[0]);
    double *min1 = kd_node_min(node[1]);
    double *max0 = kd_node_max(node[0]);
    double *max1 = kd_node_max(node[1]);
    for(d = 0; d < Nd; d++) {
        double min, max;
        double realmin, realmax;
        min = min0[d] - max1[d];
        max = max0[d] - min1[d];
        kd_realdiff(node[0]->store, min, max, &realmin, &realmax, d);
        distmin += realmin * realmin;
        distmax += realmax * realmax;
    }
    /*
    printf("%g %g %g \n", distmin, distmax, maxr * maxr);
    print(node[0]);
    print(node[1]);
    */
    int i;
    /* as Nbins are sorted, 
     * when we open the nodes
     * we know there is no need to count towards
     * other than count[start, end]
     *
     * r2[...start-1] <= distmin and
     * r2[start] > distmin
     *
     * r2[end...] >= distmax and
     * r2[end-1] < distmax
     * */
    int start = bisect_right(distmin, r2, Nbins);
    int end = bisect_left(distmax, r2, Nbins);
    for(i = end; i < Nbins; i++) {
        count[i] += node[0]->size * node[1]->size;
    }

    //printf("start , end %d %d\n", start, end);
    //printf("distmin, distmax %g %g\n", distmin, distmax);
    /* all bins are quickly counted no need to open*/
    if(start >= end) return 0;

    /* nodes may intersect, open them */
    int open = node[0]->size < node[1]->size;
    if(node[open]->dim < 0) {
        open = (open == 0);
    }
    if(node[open]->dim >= 0) {
        KDNode * save = node[open];
        node[open] = save->link[0];
        int rt;
        rt = kd_count(node, &r2[start], &count[start], end - start);
        if(rt != 0) {
            node[open] = save;
            return rt;
        }
        node[open] = save->link[1];
        rt = kd_count(node, &r2[start], &count[start], end - start);
        node[open] = save;
        return rt;
    } else {
        /* can't open the node, need to enumerate */
        return kd_count_force(node, &r2[start], &count[start], end - start);
    }
}
static int kd_count_force(KDNode * node[2], double * r2,
        uint64_t * count, size_t Nbins) {
    ptrdiff_t i, j;
    int d;
    KDStore * t0 = node[0]->store;
    KDStore * t1 = node[1]->store;
    int Nd = t0->Nd;

    double * p0base = alloca(node[0]->size * sizeof(double) * Nd);
    double * p1base = alloca(node[1]->size * sizeof(double) * Nd);
    /* collect all node[1] positions to a continue block */
    double * p1, * p0;
    double half[Nd];
    double full[Nd];
    KDEnumData endata;
    int b;

    if(t0->boxsize) {
        for(d = 0; d < Nd; d++) {
            half[d] = t0->boxsize[d] * 0.5;
            full[d] = t0->boxsize[d];
        }
    }

    kd_collect(node[0], p0base);
    kd_collect(node[1], p1base);

    for (p0 = p0base, i = 0; i < node[0]->size; i++) {
        endata.i = t0->ind[i + node[0]->start];
        for (p1 = p1base, j = 0; j < node[1]->size; j++) {
            double rr = 0.0;
            for (d = 0; d < Nd; d++){
                double dx = p1[d] - p0[d];
                if (dx < 0) dx = - dx;
                if (t0->boxsize) {
                    if (dx > half[d]) dx = full[d] - dx;
                }
                rr += dx * dx;
            }
            for(b = 0; b < Nbins; b++) {
                if(rr < r2[b]) {
                    endata.j = t1->ind[j + node[1]->start];
                    count[b] += 1;
                }
            }
            p1 += Nd;
        }
        p0 += Nd;
    }
    return 0;
    
}

#include "kd_enum.h"
#include "kd_tearoff.h"

