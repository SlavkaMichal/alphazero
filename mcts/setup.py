from setuptools import setup, Extension
from setuptools.command import build_ext
from pybind11 import get_include
from os.path import dirname
import sys
sys.path.append(dirname(sys.path[0]))
from config import SIZE, SHAPE, CPUCT, RANDOM, HEUR, DEBUG
from config import MINOR, MAJOR

print("Installing with parameters:")
print("\tsize: {}, shape: {}x{}".format(SIZE,SHAPE,SHAPE))
__version__ = "{}.{}".format(MAJOR, MINOR)

extra_compile_args = ['-std=c++14', '-Werror', '-fvisibility=hidden', ]
if HEUR:
    extra_compile_args.append('-DHEUR')
if RANDOM:
    extra_compile_args.append('-DRAND')
if DEBUG:
    extra_compile_args.append('-DDBG')

cmcts = Extension('cmcts',
        ['module.cpp','cmcts.cpp','node.cpp'],
        include_dirs = [get_include(), get_include(True)],
        language = 'c++',
        extra_compile_args = extra_compile_args,
        extra_link_args = ['-lgsl', '-lgslcblas', '-lm'],
        define_macros=[
            ('SIZE', SIZE),
            ('SHAPE', SHAPE),
            ('CPUCT', CPUCT),
            ]
        )

setup(
        name='cmcts',
        version=__version__,
        author='Michal Slavka',
#        email='slavka.michal@gmail.com',
        description='Monte Carlo Tree Search implementation for gomoku.',
        ext_modules=[ cmcts ],
        cmdclass={'build_ext' : build_ext.build_ext}
        )
