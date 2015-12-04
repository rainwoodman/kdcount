from distutils.core import setup
import numpy 

from Cython.Build import cythonize
from distutils.extension import Extension

def myext(*args):
    return Extension(*args, include_dirs=["kdcount/", numpy.get_include()])

extensions = [
        myext("kdcount.pykdcount", ["kdcount/pykdcount.pyx"])
        ]

setup(name="kdcount", version="0.3.0",
      author="Yu Feng",
      author_email="rainwoodman@gmail.com",
      description="A slower KDTree cross correlator",
      url="http://github.com/rainwoodman/kdcount",
      zip_safe=False,
      package_dir = {'kdcount': 'kdcount'},
      packages = [
        'kdcount'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
      ext_modules = cythonize(extensions)
      )

