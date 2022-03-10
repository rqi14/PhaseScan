# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'roll_ball_c',
    sources=['roll_ball_c.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[],
)))

setup(ext_modules = cythonize(Extension(
    'enlarge_image_internal_c',
    sources=['enlarge_image_internal_c.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[],
)))

setup(ext_modules=cythonize(Extension(
    'shrink_image_c',
    sources=['shrink_image_c.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[],
)))
