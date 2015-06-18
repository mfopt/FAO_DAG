import numpy
from setuptools import setup
from distutils.core import Extension
from numpy.distutils.system_info import get_info, BlasNotFoundError

define_macros = []
extra_link_args = []
extra_compile_args = ['-std=c++11']
include_dirs = [numpy.get_include(),'../','../../include/',
                '../../include/scs/include/']
library_dirs = []
libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads',
            'fftw3f_threads', 'fftw3l_threads']

blas_info = get_info('blas')
lapack_info = get_info('lapack')
if blas_info or lapack_info:
    define_macros += [('LAPACK_LIB_FOUND', None)] + blas_info.pop('define_macros', []) + lapack_info.pop('define_macros', [])
    include_dirs += blas_info.pop('include_dirs', []) + lapack_info.pop('include_dirs', [])
    library_dirs += blas_info.pop('library_dirs', []) + lapack_info.pop('library_dirs', [])
    libraries += blas_info.pop('libraries', []) + lapack_info.pop('libraries', [])
    extra_link_args += blas_info.pop('extra_link_args', []) + lapack_info.pop('extra_link_args', [])
    extra_compile_args += blas_info.pop('extra_compile_args', []) + lapack_info.pop('extra_compile_args', [])

canon = Extension('_FAO_DAG',
	sources=['FAO_DAG.i'],
	swig_opts=['-c++','-I../','-outcurrentdir'],
    define_macros=define_macros,
	include_dirs=include_dirs,
	extra_compile_args=extra_compile_args,
	extra_link_args=extra_link_args,
    extra_objects =[],#['../../include/scs/out/libscsindir.a'],
    libraries=libraries + ['scsindir'],
    library_dirs=library_dirs + ['../../include/scs/out'])

setup(
	name='FAO_DAG',
	version='0.0.1.dev1',
    author='Jack Zhu, John Miller, Paul Quigley',
    author_email='jackzhu@stanford.edu, millerjp@stanford.edu, piq93@stanford.edu',
	ext_modules=[canon],
    py_modules=['faoInterface','FAO_DAG'],
    description='A low-level library to perform the matrix building step in cvxpy, a convex optimization modeling software.',
    license='GPLv3',
    url='https://github.com/jacklzhu/FAO_DAG',
    install_requires=['numpy']
)