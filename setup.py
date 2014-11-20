from distutils.core import setup
import numpy 

from Cython.Build import cythonize
from distutils.extension import Extension

def myext(*args):
    return Extension(*args, include_dirs=["kdcount/", numpy.get_include()])

extensions = [
        myext("kdcount.pykdcount", ["kdcount/pykdcount.pyx"])
        ]

setup(name="kdcount", version="0.1",
      author="Yu Feng",
      author_email="yfeng1@andrew.cmu.edu",
      description="A slower KDTree cross correlator",
      url="http://github.com/rainwoodman/kdcount",
      download_url="http://web.phys.cmu.edu/~yfeng1/kdcount-0.1.tar.gz",
      zip_safe=False,
      package_dir = {'kdcount': 'kdcount'},
      packages = [
        'kdcount'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
      ext_modules = cythonize(extensions)
      )

