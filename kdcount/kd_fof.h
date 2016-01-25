typedef struct KDFOFData {
    ptrdiff_t * head;
    ptrdiff_t * next;
    ptrdiff_t * len;
    double ll;
    ptrdiff_t merged;
} KDFOFData;

static int _kd_fof_callback(void * data, KDEnumData * enumdata) {
    KDFOFData * fof = (KDFOFData*) data;
    ptrdiff_t i = enumdata->i;
    ptrdiff_t j = enumdata->j;

    if(enumdata->r > fof->ll) return 0;

    if(i >= j) return 0;
    if(fof->head[i] == fof->head[j]) return 0;
    fof->merged ++;

    if (fof->len[i] < fof->len[j] ) {
        /* merge in the shorter list */
        ptrdiff_t tmp;
        tmp = i;
        i = j;
        j = tmp;
    }

    /* update the length */
    fof->len[fof->head[i]] += fof->len[fof->head[j]];
    ptrdiff_t oldnext = fof->next[i];
    //printf("attaching %td to %td, oldnext = %td\n", j, i, oldnext);
    fof->next[i] = fof->head[j];

    ptrdiff_t k, kk;
    kk = fof->head[j]; /* shut-up the compiler, we know the loop will 
                          run at least once. */

    /* update the head marker of each element of the joined list */
    for(k = fof->head[j]; k >= 0 ; kk=k, k = fof->next[k]) {
        fof->head[k] = fof->head[i];
    }
    /* append items after i to the end of the merged list */
    fof->next[kk] = oldnext;
    //printf("reattaching %td to %td\n", oldnext, kk);

    return 0;
}

int kd_fof(KDNode * tree, double linking_length, ptrdiff_t * head) {
    KDNode * nodes[2] = {tree, tree};
    KDFOFData * fof = & (KDFOFData) {};

    fof->head = head;
    fof->next = malloc(sizeof(fof->next[0]) * tree->size);
    fof->len = malloc(sizeof(fof->len[0]) * tree->size);
    fof->ll = linking_length;

    ptrdiff_t i;
    for(i = 0; i < tree->size; i ++) {
        fof->head[i] = i;
        fof->next[i] = -1;
        fof->len[i] = 1;
    }
    
    int iter = 0;
    do {
        /* I am not sure how many iters we need */
        fof->merged = 0;
        kd_enum(nodes, linking_length, _kd_fof_callback, fof);
        iter ++;
        //printf("iter = %d, merged = %td\n", iter, fof->merged);
        if(iter > 10) {
            goto exc_bad;
        }
    } while(fof->merged != 0);

    free(fof->next);
    free(fof->len);
    return 0;

exc_bad:
    free(fof->next);
    free(fof->len);
    return -1;
}

