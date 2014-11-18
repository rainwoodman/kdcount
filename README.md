kdcount
=======

kdcount is a simple API for brute force pair counting, there is a C interface
and a Python interface. It uses a KDTree to prune
the K-D spatial data; for each pair within a given distance D, a callback
function is called; the user-defined callback function does the actual counting. 

It supports a periodic boundary.

The time complexity is O[(D / n) ** d] [n is number density, because each pair is opened. Note that
smarter algorithms exist (more or less, O(D / n log Dn), I may remembered it
wrong See Gary and Moore 200?, it is implemented too, though not very much tested). 
Unfortunately in cosmology we usually want to project the pair separation along
parallel + perpendicular direction relative to a given observer -- in this case,
the smarter alroithm become very difficult to implement. 

The spatial complexity is O[1] [obviously excluding the storage of the position
and data], thanks to the use of callback.

The python interface is more complicated, and powerful:
 * paircounting;
 * clustering via Friend-of-Friend algorithm;
 * multiprocessing if `sharedmem` is installed.

Python Interface of kdcount
===========================

Paircounting
-----------
kdcount takes two types of input data: 'point' and 'field'. 

'point(pos, weight, extra=None)' describes data with position and weight. For example, galaxies and
quasars are point data. 
'point.pos' is a row array of the positions of the points; other fields are
used internally.
'point.extra' is the extra properties that can be used in the Binning. One use
is to exclude the Lyman-alpha pixels and Quasars from the same sightline. 

'field(pos, value, weight, extra=None)' describes a continious field sampled at given positions, each sample
with a weight; a notorious example is the over-flux field in Lyman-alpha forest
-- it is a proxy of the over-density field sampled along quasar sightlines. 
'field.pos' is a row array of the positions of the points; other fields are used
internally.

In the Python Interface, to count, one has to define the 'binning' scheme, by
subclassing 'correlate.Binning'. 'Binning' describes a multi-dimension binning
scheme. The dimensions can be derived, for example, the norm of the spatial
separation can be a dimension the same way as the 'x' separation. We look at an
example RmuBinning that bins the norm of separation and the angle mu =
cos(theta) relative to line of sight from a given observer. 

The convention of Binning is (see numpy.digitize):
  value smaller than left boundary has bin number 0
  value bigger than right boundary has bin number Nbins + 1
Note 'numpy.digitize' uses a binary search to locate bins; 
this is slower than Binning.linear which assumes uniform bins.

class RmuBinning(Binning):
    def __init__(self, Rmax, Nbins, Nmubins, observer):
        Binning.__init__(self, 
                (0, Rmax, Nbins),
                (-1, 1, Nmubins)
            )
        self.observer = numpy.array(observer)

#  '__init__' declares the dimensions by chaining up to Binning.__init__ with a
#  list of 3-tuples describing the bins along each dimension.

    def __call__(self, r, i, j, data1, data2):
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        center = numpy.einsum('ij, ij->i', center, center) ** 0.5
        mu = dot / (center * r)
        mu[r == 0] = 10.0
        return self.linear(r, mu)

#  '__call__' calculates the bin number of pairs separated by distances r, 
#  with index (i, j). Also provides are the two input data, data1, and data2.
#  The position of first point is data1.pos[i], the position of second point is
#  data2.pos[j]. the r, mu are then converted to a linear bin index by chaining up to
#  self.linear.

Several binning schema are implemented:
   RBinning,           binning by spatial separation.
   XYBinning,          binning by X, and Y separation. X is along line of sight, Y is perpendicular
   RmuBinning,         binning by separation and angle.

The function 'paircount' counts the number of pairs according to a Binning
scheme. It returns an object with the following attributes:

        sum1 :    the numerator in the correlator
        sum2 :    the denominator in the correlator
        corr :    sum1 / sum2

        for points x points: 
               sum1 = sum( w1 w2 )
               sum2 = 1.0 
        for field x points:
               sum1 = sum( w1 w2 v1)
               sum2 = sum( w1 w2)
        for field x field:
               sum1 = sum( w1 w2 v1 v2)
               sum2 = sum( w1 w2)

        with this convention the usual form of landy-sarley
        (DD.sum1 -2r DR.sum1 + r2 RR.sum1) / (r2 RR.sum1) 
        (with r = sum(wD) / sum(wR))


        centers : the centers of the corresponding corr bin
                  centers = (X, Y, ....)
                  len(X) == corr.shape[0], len(Y) == corr.shape[1]
        binning : the binning object to create this paircount 
        edges :   the edges of the corr bins.

        fullcorr : the full correlation function with outliners 
                    len(X) == corr.shape[0] + 2 
        fullsum1 : full version of sum1
        fullsum2 : full version of sum2

Clustering
-----------
The Friend-of-Friend clustering algorithm is implemented (Davis et al. 1985 ADSlink:)
The algorithm is widely used in astronomy and cosmology simulations. All points
that are within a given 'linking_length' are clustered into one object.

clustering.fof(dataset, linkinglength):
    
    N      : number of identified objects
    labels : the object label of the data points
    sum()  : total mass of each object, for any given weight. If not given, use
            weights of dataset
    center() : center of mass of each object

