kdcount
=======

kdcount is a simple API for brute force pair counting, there is a C interface
and a Python interface. It uses a KDTree to prune
the K-D spatial data; for each pair within a given distance D, a callback
function is called; the user-defined callback function does the actual counting. 

.. image:: https://api.travis-ci.org/rainwoodman/kdcount.svg
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/kdcount/

To cite kdcount, use the DOI via Zenodo

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1051242.svg
   :target: https://doi.org/10.5281/zenodo.1051242

Periodic boundary is supported, and it is non-trivial.

The python interface is more complicated, and powerful:

 * paircounting via :py:mod:`kdcount.correlate`
 * clustering via Friend-of-Friend algorithm; :py:mod:`kdcount.cluster`
 
The calculation can be made parallel, if :py:mod:`sharedmem` is installed.

.. code:: sh

    easy_install --user sharedmem

Refer to the to the API Reference at http://rainwoodman.github.io/kdcount

The time complexity is :code:`O[(D / n) ** d]`, 
where n is number density. Each pair is opened. 

Note that
smarter algorithms exist (more or less, O(D / n log Dn), I may remembered it
wrongly See Gary and Moore 2001. 
The smarter algorithm is internally implemented, but not very much tested, and undocumented;
do not use it.

Unfortunately in cosmology we usually want to project the pair separation along
parallel + perpendicular direction relative to a given observer -- in this case,
the smarter algorithm become very difficult to implement. 

The spatial complexity is a constant, as we make extensive use of callback functions

   
