from setuptools import setup, Extension
from setuptools.command import build_ext
from pybind11 import get_include
from const import SIZE, SHAPE, CPUCT

__version__ = '0.0.1'

cmcts = Extension('cmcts',
        #['module.cpp','cmcts.h','cmcts.cpp','state.cpp','state.h'],
        ['module.cpp','cmcts.cpp','node.cpp'],
        #['cmcts.cpp'],
        include_dirs = [get_include(), get_include(True)],
        language = 'c++',
        extra_compile_args = ['-std=c++11'],
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
        email='slavka.michal@gmail.com',
        description='Monte Carlo Tree Search implementation for gomoku.',
        ext_modules=[ cmcts ],
        cmd_class={'build_ext' : build_ext.build_ext}
        )
