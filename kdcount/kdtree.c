#include "kdtree.h"

static void * 
kd_malloc(KDTree * tree, size_t size) 
{
    if(tree->malloc != NULL) {
        return tree->malloc(tree->userdata, size);
    } else {
        return malloc(size);
    }
}

static KDNode * 
kd_alloc(KDTree * tree) 
{
    KDNode * ptr = kd_malloc(tree, sizeof(KDNode) + 
            sizeof(double) * 2 * tree->input.dims[1]
            );
    ptr->link[0] = NULL;
    ptr->link[1] = NULL;
    ptr->tree = tree;
    return ptr;
}
static void
kd_build_update_min_max(KDNode * node, double min[3], double max[3])
{
    int Nd = node->tree->input.dims[1];
    int d;
    ptrdiff_t i;
    if(node->size > 0) {
        for(d = 0; d < Nd; d++) {
            max[d] = kd_input(node->tree, node->start + 0, d);
            min[d] = max[d];
        }
    } else {
        for(d = 0; d < Nd; d++) {
            max[d] = 0;
            min[d] = 0;
        }
    }
    for(i = 0; i < node->size; i++) {
        for (d = 0; d < Nd; d++) {
            double x = kd_input(node->tree, node->start + i, d);
            if (max[d] < x) max[d] = x;
            if (min[d] > x) min[d] = x;
        }
    }

}
static ptrdiff_t
kd_build_pivot(KDNode * node, double split)
{
    ptrdiff_t p, q;
    p = node->start;
    q = node->start + node->size - 1;
    while(p <= q) {
        if(kd_input(node->tree, p, node->dim) < split) {
            p ++;
        } else if(kd_input(node->tree, q, node->dim) >= split) {
            q --;
        } else {
            kd_swap(node->tree, p, q); 
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
    return p;
}
static ptrdiff_t 
kd_build_split(KDNode * node, double minhint[], double maxhint[], ptrdiff_t next) 
{
    KDTree * tree = node->tree;
    ptrdiff_t p, j;
    int d;
    int Nd = tree->input.dims[1];
    double * max = kd_node_max(node);
    double * min = kd_node_min(node);
    for(d = 0; d < Nd; d++) {
        max[d] = maxhint[d];
        min[d] = minhint[d];
    }

    if(node->size <= tree->thresh) {
        node->dim = -1;
        kd_build_update_min_max(node, min, max);
        return next;
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

    double split;

    /*
    printf("trysplit @ %g (%g %g %g %g %g %g) dim = %d, %td %td\n",
            node->split, 
            max[0], max[1], max[2], min[0],  min[1],  min[2],  
            node->dim, node->start, node->size);
    */
    p = node->start;

    /* recover from an imbalanced split */

    if(p == node->start || p == node->start + node->size) {
        /* use the middle of the box */
        split = (max[node->dim] + min[node->dim]) * 0.5;
        p = kd_build_pivot(node, split);
    }

    if(p == node->start || p == node->start + node->size) {
        /* Use the median of three samples as a split */
        double a = kd_input(tree, node->start, node->dim);
        double b = kd_input(tree, node->start + node->size / 2, node->dim);
        double c = kd_input(tree, node->start + node->size - 1, node->dim);
        /* trick from http://stackoverflow.com/a/19045659 */
        /* to prove assume a < b without loosing generality. */
        split = fmax(fmin(a, b), fmin(fmax(a, b), c));
        p = kd_build_pivot(node, split);
    }
    if(p == node->start || p == node->start + node->size) {
        /* shrink min max and retry */
        kd_build_update_min_max(node, min, max);
        split = (max[node->dim] + min[node->dim]) * 0.5;
        p = kd_build_pivot(node, split);
    }

    /* 
     * after sliding we have data[0, .... p - 1] <= split
     * and data[q +1.... end] >= split */

    /* Last resort: Now we will split at the first / last item. */
    if(p == node->start) {
        for(j = node->start + 1; j < node->start + node->size; j++) {
            if (kd_input(tree, j, node->dim) <
                kd_input(tree, node->start, node->dim)) {
                kd_swap(tree, j, node->start);
            }
        }
        split = kd_input(tree, node->start, node->dim);
        p = node->start + 1;
    }
    if(p == node->start + node->size) {
        p = node->start + node->size - 1;
        for(j = node->start; j < node->start + node->size- 1; j++) {
            if (kd_input(tree, j, node->dim) > 
                kd_input(tree, node->start + node->size - 1, node->dim)) {
                kd_swap(tree, j, node->start + node->size - 1);
            }
        }
        split = kd_input(tree, node->start + node->size - 1, node->dim);
    }

    node->split = split;

    node->link[0] = kd_alloc(tree);
    node->link[0]->index = next++;
    node->link[0]->start = node->start;
    node->link[0]->size = p - node->start;
    node->link[1] = kd_alloc(tree);
    node->link[1]->index = next++;
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
    next = kd_build_split(node->link[0], minhint, midhint, next);

    for(d = 0; d < Nd; d++) {
        midhint[d] = minhint[d];
    }
    midhint[node->dim] = node->split;
    next = kd_build_split(node->link[1], midhint, maxhint, next);

    for(d = 0; d < Nd; d++) {
        double * max1 = kd_node_max(node->link[1]);
        double * min1 = kd_node_min(node->link[1]);
        max[d] = kd_node_max(node->link[0])[d];
        if(max[d] < max1[d]) max[d] = max1[d];
        min[d] = kd_node_min(node->link[0])[d];
        if(min[d] > min1[d]) min[d] = min1[d];
    }
    return next;
}

/* 
 * create a root KDNode based on input data specified in KDTree 
 * free it with kd_free
 * */
KDNode * 
kd_build(KDTree * tree) 
{
    ptrdiff_t i;
    int Nd = tree->input.dims[1];
    double min[Nd];
    double max[Nd];
    int d;
    if (tree->ind_size > 0) {
        for(d = 0; d < Nd; d++) {
            min[d] = kd_input(tree, 0, d);
            max[d] = kd_input(tree, 0, d);
        }
    }
    else {
        for(d = 0; d < Nd; d++) {
            min[d] = 0;
            max[d] = 0;
        }
    }
    for(i = 0; i < tree->ind_size; i++) {
        for(d = 0; d < Nd; d++) {
            double data = kd_input(tree, i, d);
            if(min[d] > data) { min[d] = data; }
            if(max[d] < data) { max[d] = data; }
        }
    }
    KDNode * root = kd_alloc(tree);
    root->start = 0;
    root->index = 0;
    root->size = tree->ind_size;
    tree->size = kd_build_split(root, min, max, 1);
    return root;
}
/**
 * free a tree
 * this is recursive
 * */
void 
kd_free0(KDTree * tree, size_t size, void * ptr) 
{
    if(tree->free == NULL) {
        free(ptr);
    } else {
        tree->free(tree->userdata, size, ptr);
    }
}
void 
kd_free(KDNode * node) 
{
    if(node->link[0]) kd_free(node->link[0]);
    if(node->link[1]) kd_free(node->link[1]);
    node->tree->size --;
    kd_free0(node->tree, 
            sizeof(KDNode) +
            sizeof(double) * 2 * node->tree->input.dims[1],
            node);
}

static double * 
kd_attr_init_r(KDAttr * attr, KDNode * node)
{
    double * rt = &attr->buffer[node->index * attr->input.dims[1]];
    ptrdiff_t d;
    for(d = 0; d < attr->input.dims[1]; d++) {
        rt[d] = 0;
    }
    if (node->dim < 0) {
        ptrdiff_t i;
        for(i = node->start; i < node->start + node->size; i ++) {
            for(d = 0; d < attr->input.dims[1]; d++) {
                rt[d] += kd_attr_get(attr, i, d);
            }
        }
        return rt;
    }
     
    double * left = kd_attr_init_r(attr, node->link[0]);
    double * right = kd_attr_init_r(attr, node->link[1]);
    for(d = 0; d < attr->input.dims[1]; d++) {
        rt[d] = left[d] + right[d];
    }
    return rt;
}

void 
kd_attr_init(KDAttr * attr, KDNode * root) 
{
    kd_attr_init_r(attr, root); 
}

