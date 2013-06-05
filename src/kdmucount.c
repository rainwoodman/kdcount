#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "kdcount.h"

static double cast(double * p1, void * data) {
    return *p1;
}

static int myfree(size_t size, void * ptr) {
    free(ptr);
}

typedef struct CountData {
    double (* pos0)[3];
    double (* pos1)[3];
    size_t Nrbins;
    size_t Nmubins;
    ptrdiff_t * count;
    double rmax;
    ptrdiff_t sum;
} CountData;

static int callback(double r, ptrdiff_t i, ptrdiff_t j, CountData * data) {
    if (r == 0) return 0;
    data->sum ++;
    int rbin = r / data->rmax * data->Nrbins;
    if(rbin >= data->Nrbins) { rbin = data->Nrbins - 1; }
    double mid2 = 0.0;
    double dot = 0.0;
    double dr2 = 0.0;
    int d;
    double observer = (2070. / 1500.) - 0.5;
    for(d = 0; d < 3; d++) {
        double dx = data->pos0[i][d] - data->pos1[j][d];
        if(dx < 0) dx = -dx;
        if(dx > 0.5) dx = 1 - 0.5;
        double mid = data->pos0[i][d] + 0.5 * dx;
        if(d == 2) mid += observer;
        mid2 += mid * mid;
        dot += mid * dx;
        dr2 += dx * dx;
    }
    double mu = fabs(dot / (pow(mid2 * dr2, 0.5) ));
    int mubin = mu * data->Nmubins;
    if(mubin >= data->Nmubins) {mubin = data->Nmubins - 1;}
    data->count[rbin * data->Nmubins + mubin] ++;
    return 0;
}


KDNode * randomset(ptrdiff_t N) {
    static double boxsize[] = {1.0, 1.0, 1.0};
    double * data = malloc(sizeof(double) * N * 3);
    ptrdiff_t * ind = malloc(sizeof(ptrdiff_t) * N);

    ptrdiff_t i, d;
    for(i = 0; i < N; i ++) {
        for(d = 0; d < 3; d++) {
            data[i * 3 + d] = rand() * 1.0 / RAND_MAX;
        }
    }

    static KDType kdtype= {
        .buffer = NULL,
        .ind = NULL,
        .size = 0,
        .strides = {3 * sizeof(double), sizeof(double)},
        .elsize = sizeof(double),
        .Nd = 3,
        .thresh = 10,
        .cast = cast,
        .boxsize = boxsize,
        .malloc = malloc,
        .free = myfree,
        .userdata = NULL
    };
    kdtype.buffer = data;
    kdtype.ind = ind;
    kdtype.size = N;
    KDNode * tree = kd_build(&kdtype);
    return tree;
}

KDNode * martin() {
    ptrdiff_t N = 1199943;
    double * data = malloc(sizeof(double) * 3 * N);
    static double boxsize[3] = {1.0, 1.0, 1.0};
    ptrdiff_t * ind = malloc(sizeof(ptrdiff_t) * N);
    ptrdiff_t i, d;

    FILE * fp = fopen( "A00_hodfit.gal", "r");
    char line[4096];
    /* skip first line */
    fgets(line, 4096, fp);
    double * ptr = data;
    for(i = 0; i < N; i++) {
        fscanf(fp, "%lf %lf %lf %*lf %*lf %*lf %*lf %*lf",
                &ptr[0], &ptr[1], &ptr[2]);
        ptr += 3;
    }
    fclose(fp);

    static KDType kdtype = {
        .buffer = NULL,
        .ind = NULL,
        .size = 0,
        .strides = {3 * sizeof(double), sizeof(double)},
        .elsize = sizeof(double),
        .Nd = 3,
        .thresh = 10,
        .cast = cast,
        .boxsize = boxsize,
        .malloc = malloc,
        .free = myfree,
        .userdata = NULL
    };
    kdtype.buffer = data;
    kdtype.ind = ind;
    kdtype.size = N;
    return kd_build(&kdtype);
}

int main() {
    KDNode * R = randomset(10000);
    KDNode * D = martin();
    printf("data is ready\n");


    ptrdiff_t countRR[128][10] = {0};
    ptrdiff_t countRD[128][10] = {0};
    ptrdiff_t countDD[128][10] = {0};
    ptrdiff_t i, j;
    double rmax = 0.125;

    CountData cd = {
        .count = countRR,
        .Nrbins = 128,
        .Nmubins = 10,
        .rmax = rmax,
        .sum = 0,
    };
    cd.pos0 = R->type->buffer;
    cd.pos1 = R->type->buffer;
    KDNode * node[] = {R, R};
    kd_enum(node, cd.rmax, callback, &cd);
    printf("done with RR\n");
    cd.count = countDD;
    cd.pos0 = D->type->buffer;
    cd.pos1 = D->type->buffer;
    node[0] = D;
    node[1] = D;
    kd_enum(node, cd.rmax, callback, &cd);
    printf("done with DD\n");
    cd.count = countRD;
    cd.pos0 = R->type->buffer;
    cd.pos1 = D->type->buffer;
    node[0] = R;
    node[1] = D;
    printf("done with RD\n");
    kd_enum(node, cd.rmax, callback, &cd);
    for(j = 0; j < 10; j++) {
        for(i = 0; i < 128; i++) {
            printf("%g %g %td %td %td\n", j * 1.0 / 10, i * rmax / 128, 
                    countRR[i][j], countRD[i][j], countDD[i][j]);
        }
    }
}

