from distutils.core import setup
import numpy 

from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
        Extension("kdcount.pykdcount", [
            "kdcount/pykdcount.pyx",  
            'kdcount/kdtree.c',
            'kdcount/kd_fof.c', 'kdcount/kd_enum.c', 
            'kdcount/kd_count.c', 'kdcount/kd_integrate.c'],
            include_dirs=["kdcount/", numpy.get_include()])
        ]

setup(name="kdcount", version="0.3.4",
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

