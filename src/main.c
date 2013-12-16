#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "kdcount.h"
static double cast(double * p1) {
    return *p1;
}

KDNode * build_kdtree(double * pos, size_t length, size_t Ndim) {
    /* to free this, also need to free store, store->input, store->ind
     * they are saved in tree->store->userdata as an array (length 4)
     * free them with a for loop
     * */
    KDNode * tree = NULL;
    KD2Darray * array  = malloc(sizeof(KD2Darray));

    array->buffer = pos;
    array->dims[0] = length; 
    array->dims[1] = Ndim;
    array->strides[0] = Ndim * 8;
    array->strides[1] = 8;
    array->cast = cast;
    array->elsize = 8;

    ptrdiff_t * ind = malloc(sizeof(ptrdiff_t) * length);
    KDStore * store = malloc(sizeof(KDStore));
    store->input = array;
    store->weight = NULL;
    store->thresh = 10;
    store->ind = ind;
    store->boxsize = NULL;

    void ** ptr = malloc(sizeof(ptrdiff_t) * 8);
    store->userdata = ptr;
    ptr[0] = array;
    ptr[1] = ind;
    ptr[2] = store;
    ptr[3] = ptr;
    ptr[4] = NULL;
    return kd_build(store);
}

int main() {
    double positions[][3] = {
        {0, 0, 0}, 
        {0, 1, 1}, 
        {1, 1, 1}, 
    };
    KDNode * tree = build_kdtree(positions, 3, 3);
    return 0;
}
