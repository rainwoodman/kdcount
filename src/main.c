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
#define NXBINS 128
#define NYBINS 64

static double boxsize[] = {1.0, 1.0, 1.0};
static KDType martin = {
    .buffer = NULL,
    .ind = NULL,
    .size = 0,
    .strides = {3 * sizeof(double), sizeof(double)},
    .elsize = sizeof(double),
    .Nd = 3,
    .thresh = 10,
    .cast = (kd_castfunc) cast,
    .boxsize = boxsize,
};
static KDType noise = {
    .buffer = NULL,
    .ind = NULL,
    .size = 0,
    .strides = {3 * sizeof(double), sizeof(double)},
    .elsize = sizeof(double),
    .Nd = 3,
    .thresh = 10,
    .cast = (kd_castfunc) cast,
    .boxsize = boxsize,
};

KDNode * makenoise(ptrdiff_t N) {
    double * data = malloc(sizeof(double) * N * 3);
    ptrdiff_t * ind = malloc(sizeof(ptrdiff_t) * N);

    ptrdiff_t i, d;
    for(i = 0; i < N; i ++) {
        for(d = 0; d < 3; d++) {
            data[i * 3 + d] = rand() * 1.0 / RAND_MAX;
        }
    }

    noise.buffer = (void*)data;
    noise.ind = ind;
    noise.size = N;
    return kd_build(&noise);
}

KDNode * makemartin(ptrdiff_t N) {
    double observer = (2070. / 1500.) - 0.5;
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
        double v[3];
        fscanf(fp, "%lf %lf %lf %lf %lf %lf %*lf %*lf",
                &ptr[0], &ptr[1], &ptr[2], &v[0], &v[1], &v[2]);
        double r2 = 0.0;
        double dot = 0.0;
        for(d = 0; d < 3; d++) {
            double dx = ptr[d];
            if(d == 2) dx += observer;
            r2 += dx * dx;
            dot += dx * v[d];
        }
        for(d = 0; d < 3; d++) {
            double dx = ptr[d];
            if(d == 2) dx += observer;
            ptr[d] += dot * dx / r2;
            if(ptr[d] > 1.0) ptr[d] -= 1.0;
            if(ptr[d] < 0.0) ptr[d] += 1.0;
        }
        
        ptr += 3;
    }
    fclose(fp);

    martin.buffer = (void*)data;
    martin.ind = ind;
    martin.size = N;
    return kd_build(&martin);
}

typedef struct CountData {
    size_t Nxbins;
    size_t Nybins;
    ptrdiff_t (*count)[NYBINS];
    double rmax;
    ptrdiff_t sum;
    double (*data1)[3];
    double (*data2)[3];
} CountData;

static double eval_mu(double x[3], double y[3]) {
    double mid2 = 0.0;
    double dot = 0.0;
    double dr2 = 0.0;
    int d;
    double observer = (2070. / 1500.) - 0.5;
    for(d = 0; d < 3; d++) {
        double dx = y[d] - x[d];
        if(dx > 0.5) dx = dx - 1;
        if(dx < -0.5) dx = dx + 1;
        double mid = x[d] + 0.5 * dx;
        if(d == 2) mid += observer;
        mid2 += mid * mid;
        dot += mid * dx;
        dr2 += dx * dx;
    }
    double mu = dot / (pow(mid2 * dr2, 0.5) );
    return mu;
}

static int callback(CountData * data, double r, ptrdiff_t i, ptrdiff_t j) {
    data->sum ++;
    if(r == 0) return 0;
    double * x = data->data1[i];
    double * y = data->data2[j];

    double mu = eval_mu(x, y);
    double dx = r * mu;
    double dy = r * pow(1 - mu * mu, 0.5);
//    if(mu > 1.0 || mu < -1.0) abort();

    int xbin = (dx + data->rmax) / (2 * data->rmax) * data->Nxbins;
    if(xbin >= data->Nxbins) { return 0; }
    if(xbin < 0) { return 0; }
    int ybin = dy / data->rmax * data->Nybins;
    if(ybin >= data->Nybins) { return 0; }

    data->count[xbin][ybin] ++;
    return 0;
}


static void count_ser(KDNode * node1, KDNode * node2, double rmax, ptrdiff_t count[][NYBINS]) {
    CountData cd = {
         .count = count,
         .Nxbins = NXBINS,
         .Nybins = NYBINS,
         .rmax = rmax,
         .sum = 0,
         .data1 = node1->type->buffer,
         .data2 = node2->type->buffer,
    };
    KDNode * node[] = {node1, node2};
    kd_enum(node, cd.rmax, (kd_enum_callback) callback, &cd);
}

static void count_omp(KDNode * node1, KDNode * node2, double rmax, ptrdiff_t count[][NYBINS]) {
    ptrdiff_t len;
    KDNode ** list = kd_split(node2, node2->size / 128, &len);
    printf("len of list = %td\n", len);
#pragma omp parallel
    {
        ptrdiff_t privcount[NXBINS][NYBINS] = {0};
        int i, j;
#pragma omp for
        for(i = 0; i < len; i++) {
            count_ser(node1, list[i], rmax, privcount);
        }
#pragma omp critical
        for(j = 0; j < NXBINS; j++) {
            for(i = 0; i < NYBINS; i++) {
                count[j][i] += privcount[j][i];
            }
        }
    }
    free(list);
}
int main() {
    KDNode * R = makenoise (1000000);
    //ptrdiff_t N = 1199943;
    KDNode * D = makemartin(1199943);
    printf("data is ready\n");

    ptrdiff_t countRR[NXBINS][NYBINS] = {0};
    ptrdiff_t countRD[NXBINS][NYBINS] = {0};
    ptrdiff_t countDD[NXBINS][NYBINS] = {0};
    double rmax = 0.125;

    count_omp(R, R, rmax, countRR);
    printf("done with RR\n");
    count_omp(D, D, rmax, countDD);
    printf("done with DD\n");
    count_omp(R, D, rmax, countRD);
    printf("done with RD\n");

    int i, j;
    double fac = D->size / (1.0 * R->size);
    for(j = 0; j < NXBINS; j++) {
        for(i = 0; i < NYBINS; i++) {
            printf("%g %g %td %td %td %g\n", j * 2 * rmax / NXBINS - rmax, i * rmax / NYBINS, 
                    countRR[j][i], countRD[j][i], countDD[j][i],
                    (countDD[j][i] - 2 * countRD[j][i] * fac + countRR[j][i] * fac * fac) / (countRR[j][i] * fac * fac)
                    );
        }
    }
}

