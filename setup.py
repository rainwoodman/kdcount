from distutils.core import setup
import numpy 

from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
        Extension("kdcount.pykdcount", ["kdcount/pykdcount.pyx"], 
            depends=['kdcount/kd_fof.h', 'kdcount/kd_enum.h', 'kdcount/kd_count.h'],
            include_dirs=["kdcount/", numpy.get_include()])
        ]

setup(name="kdcount", version="0.3.3",
      author="Yu Feng",
      author_email="rainwoodman@gmail.com",
      description="A slower KDTree cross correlator",
      url="http://github.com/rainwoodman/kdcount",
      zip_safe=False,
      package_dir = {'kdcount': 'kdcount'},
      packages = [
        'kdcount', 'kdcount.tests'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
      ext_modules = cythonize(extensions),
      )

