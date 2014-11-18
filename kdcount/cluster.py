import kdcount
import numpy

class Clustering(object):
    def __init__(self, data, linking_length):
        self.data = data
        self.linking_length = linking_length
        self.next = numpy.arange(len(data), dtype='intp')

        while self.once() > 0:
            pass

    def once(self):
        tree = self.data.tree
        next = self.next
        ll = self.linking_length
        operations = 0
        for r, i, j in tree.enumiter(tree, ll):
            mask = next[i] > j
            next[i[mask]] = j[mask]
            operations += mask
        return operations
