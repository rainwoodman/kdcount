/**
 *
 * An example to use kdcount with the C-API
 *
 *   gcc capi-example.c kdcount/kdtree.c kdcount/kd_fof.c kdcount/kd_enum.c -lm
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <kdcount/kdtree.h>

ptrdiff_t *
arange(ptrdiff_t a, ptrdiff_t b)
{
    ptrdiff_t * p = (ptrdiff_t *) malloc((b - a) * sizeof(ptrdiff_t));
    ptrdiff_t i;
    for (i = a; i < b; i++) {
        p[i - a] = i;
    }
    return p;
}

double (* make_data(int Np)) [2]
{
    double (*x)[2] = (double (*)[2]) malloc(Np * sizeof(double[2]));
    int i;
    for(i = 0; i < Np; i ++) {
        x[i][0] = i * 3 / 2;
        x[i][1] = 0;
    }
    return x;
}

int main()
{

    int Np = 8;
    double (*x)[2] = make_data(Np);

    KDTree kdtree = {
        .input = {
            .buffer = (char*) x,
            .elsize = sizeof(double),
            .dims =  {Np, 2},
            .strides =  {2 * sizeof(double), 1},
            .cast = NULL,
        },
        .ind = arange(0, Np),
        .ind_size = Np,
        .thresh = 2, /* leaf size*/
        .boxsize = (double []) { 128., 128.}, /* or NULL for non-periodic box*/
    };

    KDNode * root = kd_build(&kdtree);
    ptrdiff_t * head = malloc(sizeof(ptrdiff_t) * Np);

    {
        kd_fof(root, 2.0, head);
        printf("----- long ll all connected \n");
        int i;
        for(i = 0; i < Np; i ++) {
            printf("%05.2f %05.2f is %td\n", x[i][0], x[i][1], head[i]);
        }
    }

    {
        printf("----- short ll partially connected \n");
        kd_fof(root, 1.1, head);
        int i;
        for(i = 0; i < Np; i ++) {
            printf("%05.2f %05.2f is %td\n", x[i][0], x[i][1], head[i]);
        }
    }
    free(head);
    kd_free(root);
    free(x);
    return 0;
}
