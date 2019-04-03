import os
import sys
import subprocess

from config import *
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

print("Installing with parameters:")
print("\tsize: {}, shape: {}x{}".format(SIZE,SHAPE,SHAPE))
__version__ = "{}.{}".format(MAJOR, MINOR)
ext_name = 'cmcts'

class MCTSExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class MCTSBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                '-DPYTHON_EXECUTABLE=' + sys.executable,
                '-DSHAPE={}'.format(SHAPE),
                '-DSIZE={}' .format(SIZE),
                '-DMAJOR={}'.format(MAJOR),
                '-DMINOR={}'.format(MINOR),
                '-DLOCAL_SITE_PATH={}'.format(LOCAL_SITE_PATH),
                '-DPREFIX={}'.format(PREFIX),
                '-Wno-dev',
                ]

        if HEUR:
            cmake_args.append('-DHEUR=ON')
        if EXTENSION:
            cmake_args.append('-DEXT=ON')
        if DEBUG:
            cmake_args.append('-DDEBUG=ON')
        if CUDA:
            cmake_args.append('-DCUDA=ON')

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(ext.sourcedir)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
        name=ext_name,
        version=__version__,
        author='Michal Slavka',
        author_email='slavka.michal@gmail.com',
        description='Monte Carlo Tree Search implementation for gomoku.',
        long_description='',
        ext_modules=[ MCTSExtension(ext_name) ],
        cmdclass=dict(build_ext=MCTSBuild),
        zip_safe=False
        )
