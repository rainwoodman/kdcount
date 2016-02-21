#include "kdtree.h"

typedef struct KDIntegrateData {
    KDAttr * attr;
    uint64_t* count;
    double * weight;
    double * min;
    double * max;
    int Nw;
} KDIntegrateData;

static void 
kd_integrate_check(KDIntegrateData * kdid, KDNode * node) 
{

    ptrdiff_t i, j;
    int d;
    KDTree * t0 = node->tree;
    int Nd = t0->input.dims[1];
    int Nw = kdid->Nw;

    double * p0base = alloca(node->size * sizeof(double) * Nd);
    double * w0base = alloca(node->size * sizeof(double) * Nw);
    /* collect all node positions to a continue block */
    double *p0, *w0;

    kd_collect(node, &t0->input, p0base);
    if(Nw > 0) {
        kd_collect(node, &kdid->attr->input, w0base);
    }

    for (p0 = p0base, w0 = w0base, i = 0; i < node->size; i++) {
        int inside = 1;
        for (d = 0; d < Nd; d++){
            if(p0[d] < kdid->min[d] || p0[d] >= kdid->max[d]) {
                inside = 0;
                break;
            }
        }
        if(inside) {
            kdid->count[0] += 1;
            for(d = 0; d < Nw; d++) {
                kdid->weight[d] += w0[d];
            }
        }
        w0 += Nw;
        p0 += Nd;
    }
}


static void 
kd_integrate_traverse(KDIntegrateData * kdid, KDNode * node) 
{
    int Nd = node->tree->input.dims[1];
    int Nw = kdid->Nw;
    int d;
    double *min0 = kd_node_min(node);
    double *max0 = kd_node_max(node);
    for(d = 0; d < Nd; d++) {
        if(min0[d] >= kdid->max[d] || max0[d] < kdid->min[d]) {
            /* fully outside, skip this node */
            return;
        }
    }

    int inside = 1;
    for(d = 0; d < Nd; d++) {
        if(min0[d] < kdid->min[d] || max0[d] >= kdid->max[d]) {
            inside = 0;
            break;
        }
    }
    if(inside) {
        /* node inside integration range */
        kdid->count[0] += node->size;
        if(Nw > 0) {
            double * w0 = kd_attr_get_node(kdid->attr, node);
            for(d = 0; d < Nw; d++) {
                kdid->weight[d] += w0[d];
            }
        }
        return;
    }

    if(node->dim < 0) {
        /* can't open the node, need to enumerate */
        kd_integrate_check(kdid, node);
    } else {
        kd_integrate_traverse(kdid, node->link[0]);
        kd_integrate_traverse(kdid, node->link[1]);
    } 
}

void 
kd_integrate(KDNode * node, KDAttr * attr, 
        uint64_t * count, double * weight, 
        double * min, double * max) 
{
    int Nw;
    int d;

    if (attr) 
        Nw = attr->input.dims[1];
    else
        Nw = 0;
    
    KDIntegrateData kdid = {
        .attr = attr,
        .count = count,
        .weight = weight,
        .Nw = Nw,
        .min = min,
        .max = max,
    };

    count[0] = 0;
    if(Nw > 0) {
        for(d = 0; d < Nw; d++) {
            weight[d] = 0;
        }
    }
    
    kd_integrate_traverse(&kdid, node);
}
