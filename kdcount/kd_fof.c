#include "kdtree.h"

typedef struct TraverseData {
    ptrdiff_t * head;
    ptrdiff_t * next;
    ptrdiff_t * len;
    double ll;
    ptrdiff_t merged;
} TraverseData;

static ptrdiff_t get_root(TraverseData * d, ptrdiff_t i)
{
    int r = i;
    while(d->head[r] != r) {
        r = d->head[r];
    }
    d->head[i] = r;
    return r;
}

static int
_kd_fof_callback(void * data, KDEnumPair * pair) 
{
    TraverseData * trav = (TraverseData*) data;
    ptrdiff_t i = pair->i;
    ptrdiff_t j = pair->j;

    if(pair->r > trav->ll) return 0;

    if(i >= j) return 0;
    if (trav->len[i] < trav->len[j] ) {
        /* merge in the shorter list, j*/
        ptrdiff_t tmp;
        tmp = i;
        i = j;
        j = tmp;
    }
    ptrdiff_t root_i = get_root(trav, i);
    ptrdiff_t root_j = get_root(trav, j);

    if(root_i == root_j) return 0;

    trav->merged ++;

    /* update the length */
    trav->len[root_i] += trav->len[root_j];

    if(trav->len[root_j] < 100) {
        /* merge all as direct children of the root */
        ptrdiff_t k, kk;
        kk = 0;

        /* update the head marker of each element of the joined list */
        for(k = root_j; k >= 0 ; kk=k, k = trav->next[k]) {
            trav->head[k] = root_i;
        }

        /* maintain next array only if the new proto-halo is short */
        if(trav->len[root_i] < 100) {
            /* append items after i to the end of the merged list */
            ptrdiff_t oldnext = trav->next[root_i];
            //printf("attaching %td to %td, oldnext = %td\n", j, i, oldnext);
            trav->next[root_i] = root_j;
            trav->next[kk] = oldnext;
            //printf("reattaching %td to %td\n", oldnext, kk);
        }
    } else {
        /* We will never need to traverse the children for halos > 10,
         *  so do not try to maintain it */

        /* merge root_j as direct subtree of the root */
        trav->head[root_j] = root_i;
    }
    return 0;
}

int 
kd_fof(KDNode * tree, double linking_length, ptrdiff_t * head) 
{
    KDNode * nodes[2] = {tree, tree};
    TraverseData * trav = & (TraverseData) {};

    trav->head = head;
    trav->next = malloc(sizeof(trav->next[0]) * tree->size);
    trav->len = malloc(sizeof(trav->len[0]) * tree->size);
    trav->ll = linking_length;

    ptrdiff_t i;
    for(i = 0; i < tree->size; i ++) {
        trav->head[i] = i;
        trav->next[i] = -1;
        trav->len[i] = 1;
    }
    
    int iter = 0;
    do {
        /* I am not sure how many iters we need */
        trav->merged = 0;
        kd_enum(nodes, linking_length, _kd_fof_callback, trav);
        for(i = 0; i < tree->size; i ++) {
            trav->head[i] = get_root(trav, i);
        }
        iter ++;
        //printf("iter = %d, merged = %td\n", iter, trav->merged);
        if(iter > 10) {
            goto exc_bad;
        }
    } while(trav->merged != 0);

    free(trav->next);
    free(trav->len);
    return 0;

exc_bad:
    free(trav->next);
    free(trav->len);
    return -1;
}

