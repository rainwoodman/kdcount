static int kd_enum_check(KDNode * nodes[2], double rmax2,
        kd_enum_callback callback, void * data);
/*
 * enumerate two KDNode trees, up to radius max.
 *
 * for each pair i in nodes[0] and j in nodes[1],
 * if the distance is smaller than maxr,
 * call callback.
 * if callback returns nonzero, terminate and return the value
 * */
int kd_enum(KDNode * nodes[2], double maxr,
        kd_enum_callback callback, void * data) {
    int Nd = nodes[0]->tree->input.dims[1];
    double distmax = 0, distmin = 0;
    double rmax2 = maxr * maxr;
    int d;
    double *min0 = kd_node_min(nodes[0]);
    double *min1 = kd_node_min(nodes[1]);
    double *max0 = kd_node_max(nodes[0]);
    double *max1 = kd_node_max(nodes[1]);
    for(d = 0; d < Nd; d++) {
        double min, max;
        double realmin, realmax;
        min = min0[d] - max1[d];
        max = max0[d] - min1[d];
        kd_realdiff(nodes[0]->tree, min, max, &realmin, &realmax, d);
        distmin += realmin * realmin;
        distmax += realmax * realmax;
    }
    /*
    printf("%g %g %g \n", distmin, distmax, maxr * maxr);
    print(nodes[0]);
    print(nodes[1]);
    */
    if (distmin > rmax2 * 1.00001) {
        /* nodes are too far, skip them */
        return 0;
    }
    if (distmax >= rmax2) {
        /* nodes may intersect, open them */
        int open = nodes[0]->size < nodes[1]->size;
        if(nodes[open]->dim < 0) {
            open = (open == 0);
        }
        if(nodes[open]->dim >= 0) {
            KDNode * save = nodes[open];
            nodes[open] = save->link[0];
            int rt;
            rt = kd_enum(nodes, maxr, callback, data);
            if(rt != 0) {
                nodes[open] = save;
                return rt;
            }
            nodes[open] = save->link[1];
            rt = kd_enum(nodes, maxr, callback, data);
            nodes[open] = save;
            return rt;
        } else {
            /* can't open the nodes, need to enumerate */
        }
    } else {
        /* fully inside, fall through,
         * and enumerate  */
    }

    return kd_enum_check(nodes, rmax2, callback, data);
}

static int kd_enum_check(KDNode * nodes[2], double rmax2,
        kd_enum_callback callback, void * data) {
    int rt = 0;
    ptrdiff_t i, j;
    int d;
    KDTree * t0 = nodes[0]->tree;
    KDTree * t1 = nodes[1]->tree;
    int Nd = t0->input.dims[1];

    double * p0base = malloc(nodes[0]->size * sizeof(double) * Nd);
    double * p1base = malloc(nodes[1]->size * sizeof(double) * Nd);
    /* collect all nodes[1] positions to a continue block */
    double * p1, * p0;
    double half[Nd];
    double full[Nd];
    KDEnumData endata;

    if(t0->boxsize) {
        for(d = 0; d < Nd; d++) {
            half[d] = t0->boxsize[d] * 0.5;
            full[d] = t0->boxsize[d];
        }
    }

    /* no need to collect weight */
    kd_collect(nodes[0], &t0->input, p0base);
    kd_collect(nodes[1], &t1->input, p1base);

    for (p0 = p0base, i = nodes[0]->start; 
        i < nodes[0]->start + nodes[0]->size; i++) {
        endata.i = t0->ind[i];
        for (p1 = p1base, j = nodes[1]->start; 
             j < nodes[1]->start + nodes[1]->size; j++) {
            double r2 = 0.0;
            for (d = 0; d < Nd; d++){
                double dx = p1[d] - p0[d];
                if (dx < 0) dx = - dx;
                if (t0->boxsize) {
                    if (dx > half[d]) dx = full[d] - dx;
                }
                r2 += dx * dx;
            }
            if(r2 <= rmax2) {
                endata.j = t1->ind[j];
                endata.r = sqrt(r2);
                if(0 != callback(data, &endata)) {
                    rt = -1;
                    goto exit;
                }
            }
            p1 += Nd;
        }
        p0 += Nd;
    }
exit:
    free(p1base);
    free(p0base);
    return rt;
}

