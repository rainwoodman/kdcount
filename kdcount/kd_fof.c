#include "kdtree.h"

/* Friend of Friend:
 *
 * Connected component via edge enumeration.
 *
 * The connected components are stored as trees.
 *
 * (visit) two vertices i, j connected by an edge.
 *         if splay(i) differ from splay(j), the components shall be
 *         merged, call merge(i, j)
 *
 * (merge) Join two trees if they are connected by adding the root
 *         of i as a child of root of j.
 *
 * (splay) Move a leaf node to the direct child of the tree root
 *         and returns the root.
 *
 * One can show this algorithm ensures splay(i) is an label of
 * max connected components in the graph.
 *
 * Suitable for application where finding edges of a vertice is more expensive
 * than enumerating over edges.
 *
 * In FOF, we use the dual tree algorithm for edge enumeration.
 *
 * The storage is O(N) for the output labels.
 *
 * */

typedef struct TraverseData {
    ptrdiff_t * head;
    double ll;
    ptrdiff_t merged;
} TraverseData;

static ptrdiff_t splay(TraverseData * d, ptrdiff_t i)
{
    int r = i;
    while(d->head[r] != r) {
        r = d->head[r];
    }
    /* link the node directly to the root to keep the tree flat */
    d->head[i] = r;
    return r;
}

static int
_kd_fof_visit_edge(void * data, KDEnumPair * pair) 
{
    TraverseData * trav = (TraverseData*) data;
    ptrdiff_t i = pair->i;
    ptrdiff_t j = pair->j;

    if(pair->r > trav->ll) return 0;

    if(i >= j) return 0;

    ptrdiff_t root_i = splay(trav, i);
    ptrdiff_t root_j = splay(trav, j);

    if(root_i == root_j) return 0;

    trav->merged ++;

    /* merge root_j as direct subtree of the root */
    trav->head[root_j] = root_i;

    return 0;
}

int 
kd_fof(KDNode * tree, double linking_length, ptrdiff_t * head) 
{
    KDNode * nodes[2] = {tree, tree};
    TraverseData * trav = & (TraverseData) {};

    trav->head = head;
    trav->ll = linking_length;

    ptrdiff_t i;
    for(i = 0; i < tree->size; i ++) {
        trav->head[i] = i;
    }

    trav->merged = 0;

    kd_enum(nodes, linking_length, _kd_fof_visit_edge, trav);
    for(i = 0; i < tree->size; i ++) {
        trav->head[i] = splay(trav, i);
    }

    return 0;

exc_bad:
    return -1;
}

