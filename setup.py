from numpy.distutils.core import setup, Extension
from numpy import get_include
import monkey
setup(name="kdcount", version="0.1",
      author="Yu Feng",
      author_email="yfeng1@andrew.cmu.edu",
      description="A slower KDTree cross correlator",
      url="http://github.com/rainwoodman/kdcount",
      download_url="http://web.phys.cmu.edu/~yfeng1/kdcount-0.1.tar.gz",
      zip_safe=False,
      package_dir = {'kdcount': 'src'},
      packages = [
        'kdcount'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
      ext_modules = [
        Extension('kdcount.' + name, 
             [ 'src/' + name.replace('.', '/') + '.pyx',],
             extra_compile_args=['-O3'],
             libraries=[],
             include_dirs=[get_include(), 'src/'],
             depends = extra
        ) for name, extra in [
         ('pykdcount', ['kdcount.h']),
        ]
      ])

