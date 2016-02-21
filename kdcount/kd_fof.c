#include "kdtree.h"

typedef struct TraverseData {
    ptrdiff_t * head;
    ptrdiff_t * next;
    ptrdiff_t * len;
    double ll;
    ptrdiff_t merged;
} TraverseData;

static int 
_kd_fof_callback(void * data, KDEnumPair * pair) 
{
    TraverseData * trav = (TraverseData*) data;
    ptrdiff_t i = pair->i;
    ptrdiff_t j = pair->j;

    if(pair->r > trav->ll) return 0;

    if(i >= j) return 0;
    if(trav->head[i] == trav->head[j]) return 0;
    trav->merged ++;

    if (trav->len[i] < trav->len[j] ) {
        /* merge in the shorter list */
        ptrdiff_t tmp;
        tmp = i;
        i = j;
        j = tmp;
    }

    /* update the length */
    trav->len[trav->head[i]] += trav->len[trav->head[j]];
    ptrdiff_t oldnext = trav->next[i];
    //printf("attaching %td to %td, oldnext = %td\n", j, i, oldnext);
    trav->next[i] = trav->head[j];

    ptrdiff_t k, kk;
    kk = trav->head[j]; /* shut-up the compiler, we know the loop will 
                          run at least once. */

    /* update the head marker of each element of the joined list */
    for(k = trav->head[j]; k >= 0 ; kk=k, k = trav->next[k]) {
        trav->head[k] = trav->head[i];
    }
    /* append items after i to the end of the merged list */
    trav->next[kk] = oldnext;
    //printf("reattaching %td to %td\n", oldnext, kk);

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

