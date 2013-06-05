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
    int Nd = node[0]->store->input.dims[1];
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
    int Nd = t0->input.dims[1];

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

