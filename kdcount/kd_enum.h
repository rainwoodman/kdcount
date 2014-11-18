static int kd_enum_force(KDNode * node[2], double rmax2,
        kd_enum_callback callback, void * data);
/*
 * enumerate two KDNode trees, up to radius max.
 *
 * for each pair i in node[0] and j in node[1],
 * if the distance is smaller than maxr,
 * call callback.
 * if callback returns nonzero, terminate and return the value
 * */
int kd_enum(KDNode * node[2], double maxr,
        kd_enum_callback callback, void * data) {
    int Nd = node[0]->store->input.dims[1];
    double distmax = 0, distmin = 0;
    double rmax2 = maxr * maxr;
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
    if (distmin > rmax2 * 1.00001) {
        /* nodes are too far, skip them */
        return 0;
    }
    if (distmax >= rmax2) {
        /* nodes may intersect, open them */
        int open = node[0]->size < node[1]->size;
        if(node[open]->dim < 0) {
            open = (open == 0);
        }
        if(node[open]->dim >= 0) {
            KDNode * save = node[open];
            node[open] = save->link[0];
            int rt;
            rt = kd_enum(node, maxr, callback, data);
            if(rt != 0) {
                node[open] = save;
                return rt;
            }
            node[open] = save->link[1];
            rt = kd_enum(node, maxr, callback, data);
            node[open] = save;
            return rt;
        } else {
            /* can't open the node, need to enumerate */
        }
    } else {
        /* fully inside, fall through,
         * and enumerate  */
    }

    return kd_enum_force(node, rmax2, callback, data);
}

static int kd_enum_force(KDNode * node[2], double rmax2,
        kd_enum_callback callback, void * data) {
    int rt = 0;
    ptrdiff_t i, j;
    int d;
    KDStore * t0 = node[0]->store;
    KDStore * t1 = node[1]->store;
    int Nd = t0->input.dims[1];

    double * p0base = malloc(node[0]->size * sizeof(double) * Nd);
    double * p1base = malloc(node[1]->size * sizeof(double) * Nd);
    /* collect all node[1] positions to a continue block */
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
    kd_collect(node[0], p0base, NULL);
    kd_collect(node[1], p1base, NULL);

    double bad = rmax2 * 2 + 1;
    for (p0 = p0base, i = 0; i < node[0]->size; i++) {
        endata.i = t0->ind[i + node[0]->start];
        for (p1 = p1base, j = 0; j < node[1]->size; j++) {
            double r2 = 0.0;
            for (d = 0; d < Nd; d++){
                double dx = p1[d] - p0[d];
                if (dx < 0) dx = - dx;
                if (t0->boxsize) {
                    if (dx > half[d]) dx = full[d] - dx;
                }
                /*
                if (dx > maxr) {
                    r2 = bad;
                    p1 += Nd - d;
                    break;
                } */
                r2 += dx * dx;
            }
            if(r2 <= rmax2) {
                endata.j = t1->ind[j + node[1]->start];
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

