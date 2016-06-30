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
 * In KD-Tree implementation we apply optimization for overdense regions
 *
 * Before FOF we connect all nodes that are less than linking length in size (internally connected)
 * 
 * (connect) connect all points in a node into a tree, the first particle is the root.
 * 
 * In FOF, we use the dual tree algorithm for edge enumeration, but if two nodes are minimaly sperated
 * by linking length and both are internally connected
 *
 *   (nodemerge) merge(node1->first, node2->first)
 * 
 * For typically low resolution cosmological simulations this optimization improves speed by a few percent.
 * The improvement increases for highly clustered data.
 *
 * The storage is O(N) for the output labels + O(M) for the connection status of kdtree nodes.
 *
 * */

typedef struct TraverseData {
    ptrdiff_t * head;
    ptrdiff_t * ind; /* tree->ind */
    char * node_connected;
    double ll;
    double ll2;
    ptrdiff_t merged;
    ptrdiff_t skipped;
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
    ptrdiff_t r = trav->ind[node->start];
    if(node->size == 1) {
        return r;
    }
    ptrdiff_t i;
    for(i = node->start + 1; i < node->size + node->start; i ++) {
        trav->head[trav->ind[i]] = r;
    }

    return r;
}

static int
_kd_fof_visit_edge(void * data, KDEnumPair * pair);

static int
_kd_fof_check_nodes(void * data, KDEnumNodePair * pair)
{
    TraverseData * trav = (TraverseData*) data;

    if(trav->node_connected[pair->nodes[0]->index]
    && trav->node_connected[pair->nodes[1]->index])
    {
        /* two fully connected nodes are linked, simply link the first particle.  */
        KDEnumPair epair;
        epair.r = sqrt(pair->distmin2);
        epair.i = trav->ind[pair->nodes[0]->start];
        epair.j = trav->ind[pair->nodes[1]->start];
        trav->skipped += pair->nodes[0]->size * pair->nodes[1]->size;
        return _kd_fof_visit_edge(data, &epair);
    } else {
        return kd_enum_check(pair->nodes, trav->ll2, _kd_fof_visit_edge, data);
    }

}

static int
_kd_fof_visit_edge(void * data, KDEnumPair * pair) 
{
    TraverseData * trav = (TraverseData*) data;
    ptrdiff_t i = pair->i;
    ptrdiff_t j = pair->j;

    if(pair->r > trav->ll) return 0;

    trav->merged ++;

    if(i >= j) return 0;

    ptrdiff_t root_i = splay(trav, i);
    ptrdiff_t root_j = splay(trav, j);

    if(root_i == root_j) return 0;

    /* merge root_j as direct subtree of the root */
    trav->head[root_j] = root_i;

    return 0;
}

static void
connect_nodes_r(TraverseData * trav, KDNode * node, int parent_connected)
{
    int d;
    double dist = 0;
    if(!parent_connected) {
        for(d = 0; d < 3; d ++) {
            double dx = kd_node_max(node)[d] - kd_node_min(node)[d];
            dist += dx * dx;
        }
        if(dist <= trav->ll2) {
            connect_node(trav, node);
            trav->node_connected[node->index] = 1;
        }
    } else {
        trav->node_connected[node->index] = 1;
    }
    if(node->dim != -1) {
        connect_nodes_r(trav, node->link[0], trav->node_connected[node->index]);
        connect_nodes_r(trav, node->link[1], trav->node_connected[node->index]);
    }
}

int 
kd_fof(KDNode * node, double linking_length, ptrdiff_t * head)
{
    KDNode * nodes[2] = {node, node};
    TraverseData * trav = & (TraverseData) {};

    trav->head = head;
    trav->ll = linking_length;
    trav->ll2 = linking_length * linking_length;
    trav->node_connected = malloc(node->tree->size);
    trav->ind = node->tree->ind;

    ptrdiff_t i;
    for(i = 0; i < node->tree->size; i ++) {
        trav->node_connected[i] = 0;
    }

    for(i = 0; i < node->size; i ++) {
        trav->head[i] = i;
    }
    size_t connected = 0;

    connect_nodes_r(trav, node, 0);

    for(i = 0; i < node->tree->size; i ++) {
        connected += trav->node_connected[i];
    }

    trav->merged = 0;
    trav->skipped = 0;

    kd_enum(nodes, linking_length, NULL, _kd_fof_check_nodes, trav);

    for(i = 0; i < node->size; i ++) {
        trav->head[i] = splay(trav, i);
    }
    //printf("skipped = %td merged = %td connected = %td size = %td\n", trav->skipped, trav->merged, connected, node->tree->size);

    free(trav->node_connected);
    return 0;

}

