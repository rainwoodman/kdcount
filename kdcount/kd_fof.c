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
 * In FOF, we use the dual tree algorithm for edge enumeration. We prune the dual tree
 * walking algorithm
 *
 * (connect) if not connected, connect all points in a node into a tree, returns
 *           the root.
 * (prune) if two nodes are maxiumimly separated by linking length, connect(node1) and connect(node2)
 *         then connect the two subtrees containing node1 and node2. Skip the edge enumeration between node1 and node2.
 *
 * The storage is O(N) for the output labels.
 *
 * */

typedef struct TraverseData {
    ptrdiff_t * head;
    char * node_connected;
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

static ptrdiff_t connect_node(TraverseData * trav, KDNode * node)
{
    KDTree * tree = node->tree;
    ptrdiff_t r = splay(trav, tree->ind[node->start]);
    if(node->size == 1) {
        return r;
    }
    if(trav->node_connected[node->index]) {
        return r;
    }
    /* fresh node, connect it */
    ptrdiff_t i;
    for(i = node->start + 1; i < node->size + node->start; i ++) {
        trav->head[tree->ind[i]] = r;
    }
    trav->node_connected[node->index] = 1;

    return r;
}

static int
_kd_fof_prune_nodes(void * data, KDEnumNodePair * pair, int * open)
{
    TraverseData * trav = (TraverseData*) data;

    if(pair->distmax2 > trav->ll * trav->ll) {
        *open = 1;
        return 0;
    }

    *open = 0;
    ptrdiff_t r1 = connect_node(trav, pair->nodes[0]);
    ptrdiff_t r2 = connect_node(trav, pair->nodes[1]);

    if(r1 != r2)
        trav->head[r2] = r1;
    return 0;
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
kd_fof(KDNode * node, double linking_length, ptrdiff_t * head)
{
    KDNode * nodes[2] = {node, node};
    TraverseData * trav = & (TraverseData) {};

    trav->head = head;
    trav->ll = linking_length;
    trav->node_connected = malloc(node->tree->size);

    ptrdiff_t i;
    for(i = 0; i < node->tree->size; i ++) {
        trav->node_connected = 0;
    }

    for(i = 0; i < node->size; i ++) {
        trav->head[i] = i;
    }

    trav->merged = 0;

    kd_enum(nodes, linking_length, _kd_fof_visit_edge, _kd_fof_prune_nodes, trav);

    for(i = 0; i < node->size; i ++) {
        trav->head[i] = splay(trav, i);
    }

    free(trav->node_connected);
    return 0;

}

