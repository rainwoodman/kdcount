static void kd_tearoff_r(KDNode * node, ptrdiff_t thresh, KDNode *** out, ptrdiff_t * length, ptrdiff_t * size);

/* this tearoffs a kdtree to a series of nodes that are no more 
 * bigger than thresh */
KDNode ** kd_tearoff(KDNode * node, ptrdiff_t thresh, ptrdiff_t * length) {
    ptrdiff_t size = 128;
    KDNode ** out = kd_malloc(node->store, sizeof(void*) * size);
    * length = 0;
    kd_tearoff_r(node, thresh, &out, length, &size);
    KDNode ** ret = kd_malloc(node->store, sizeof(void*) * *length);
    ptrdiff_t i;
    for(i = 0; i < * length; i++) {
        ret[i] = out[i]; 
    }
    kd_free0(node->store, sizeof(void*) * size, out);
    return ret;
}

static void kd_tearoff_r(KDNode * node, ptrdiff_t thresh, KDNode *** out, ptrdiff_t * length, ptrdiff_t * size) {

    if(node->size <= thresh || node->dim == -1) {
        if (*length >= *size) {
            KDNode ** old = out[0];
            out[0] = kd_malloc(node->store, sizeof(void*) * *size * 2);
            ptrdiff_t i;
            for(i = 0; i < *length; i++) {
                out[0][i] = old[i]; 
            }
            kd_free0(node->store, *size * sizeof(void*), old);
            *size *= 2;
        }
        out[0][*length] = node;
        *length = *length + 1;
        return;
    }
    kd_tearoff_r(node->link[0], thresh, out, length, size);
    kd_tearoff_r(node->link[1], thresh, out, length, size);
}
