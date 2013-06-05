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

typedef struct KD2Darray {
    /* the buffer holding array elements required */
    char * buffer; 
    /* number of points. required*/
    ptrdiff_t dims[2];
    /* the byte offset of the axes.  required
     * the i-th position , d-th component is
     * at i * strides[0] + d * strides[1] */
    ptrdiff_t strides[2]; 

    /* if cast p1 to double and return it */
    double (* cast)(void * p1);
    /* the byte size of each scalar, required */
    ptrdiff_t elsize;

} KD2Darray;

typedef struct KDStore {

/* defining the input positions */
    KD2Darray input;
/* defining the input weights */
    KD2Darray weight;

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
    KDNode * ptr = kd_malloc(store, sizeof(KDStore) + sizeof(double) * 2 * store->input.dims[1]);
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
    return kd_node_max(node) + node->store->input.dims[1];
}
static inline double kd_array_get(KD2Darray * array, ptrdiff_t i, ptrdiff_t d) {
    char * ptr = & array->buffer[
                        i * array->strides[0] + 
                        d * array->strides[1]];
    if(array->cast) {
        return array->cast(ptr);
    } else {
        return * (double*) ptr;
    }
}

static inline double kd_weight(KDStore * store, ptrdiff_t i, ptrdiff_t d) {
    i = store->ind[i];
    return kd_array_get(&store->weight, i, d);
}
static inline double kd_input(KDStore * store, ptrdiff_t i, ptrdiff_t d) {
    i = store->ind[i];
    return kd_array_get(&store->input, i, d);
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

    int Nd = store->input.dims[1];
    double * max = kd_node_max(node);
    double * min = kd_node_min(node);
    for(d = 0; d < Nd; d++) {
        max[d] = maxhint[d];
        min[d] = minhint[d];
    }

    if(node->size <= store->thresh) {
        node->dim = -1;
        return;
    }

    node->dim = 0;
    double longest = maxhint[0] - minhint[0];
    for(d = 1; d < Nd; d++) {
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
        if(kd_input(store, p, node->dim) < node->split) {
            p ++;
        } else if(kd_input(store, q, node->dim) >= node->split) {
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
            if (kd_input(store, j, node->dim) <
                kd_input(store, node->start, node->dim)) {
                kd_swap(store, j, node->start);
            }
        }
        node->split = kd_input(store, node->start, node->dim);
        p = q + 1;
    }
    if(p == node->start + node->size) {
        p = node->start + node->size - 1;
        for(j = node->start; j < node->start + node->size- 1; j++) {
            if (kd_input(store, j, node->dim) > 
                kd_input(store, node->start + node->size - 1, node->dim)) {
                kd_swap(store, j, node->start + node->size - 1);
            }
        }
        node->split = kd_input(store, node->start + node->size - 1, node->dim);
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
    double midhint[Nd];
    for(d = 0; d < Nd; d++) {
        midhint[d] = maxhint[d];
    }
    midhint[node->dim] = node->split;
    kd_build_split(node->link[0], minhint, midhint);
    for(d = 0; d < Nd; d++) {
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
    int Nd = store->input.dims[1];
    double min[Nd];
    double max[Nd];    
    int d;
    store->total_nodes = 0;
    store->ind[0] = 0;
    for(d = 0; d < Nd; d++) {
        min[d] = kd_input(store, 0, d);
        max[d] = kd_input(store, 0, d);
    }
    for(i = 0; i < store->input.dims[0]; i++) {
        store->ind[i] = i;
        for(d = 0; d < Nd; d++) {
            double data = kd_input(store, i, d);
            if(min[d] > data) { min[d] = data; }
            if(max[d] < data) { max[d] = data; }
        }
    }
    KDNode * tree = kd_alloc(store);
    tree->start = 0;
    tree->size = store->input.dims[0];
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
    kd_free0(node->store, sizeof(KDNode) + node->store->input.elsize, node);
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
    KD2Darray * array = &t->input;
    int Nd = array->dims[1];
    char * base = array->buffer;
    for (j = 0; j < node->size; j++) {
        char * item = base + t->ind[j + node->start] * array->strides[0];
        if(array->cast) {
            for(d = 0; d < Nd; d++) {
                *ptr = array->cast(item);
                ptr++;
                item += array->strides[1];
            }
        } else {
            for(d = 0; d < Nd; d++) {
                memcpy(ptr, item, array->elsize);
                ptr++;
                item += array->strides[1];
            }
        
        }
    }
}

#include "kd_count.h"
#include "kd_enum.h"
#include "kd_tearoff.h"

