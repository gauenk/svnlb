#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
print("CWD",os.getcwd())
shutil.rmtree("pyvnlb", ignore_errors=True)
os.mkdir("pyvnlb")
shutil.copytree("pylib", "pyvnlb/pylib")
shutil.copyfile("__init__.py", "pyvnlb/__init__.py")
shutil.copyfile("loader.py", "pyvnlb/loader.py")

ext = ".pyd" if platform.system() == 'Windows' else ".so"
prefix = "Release/" * (platform.system() == 'Windows')

swigpyvnlb_generic_lib = f"{prefix}_swigpyvnlb{ext}"
swigpyvnlb_avx2_lib = f"{prefix}_swigpyvnlb_avx2{ext}"

found_swigpyvnlb_generic = os.path.exists(swigpyvnlb_generic_lib)
found_swigpyvnlb_avx2 = os.path.exists(swigpyvnlb_avx2_lib)

assert (found_swigpyvnlb_generic or found_swigpyvnlb_avx2), \
    f"Could not find {swigpyvnlb_generic_lib} or " \
    f"{swigpyvnlb_avx2_lib}. Pyvnlb may not be compiled yet."

if found_swigpyvnlb_generic:
    print(f"Copying {swigpyvnlb_generic_lib}")
    shutil.copyfile("swigpyvnlb.py", "pyvnlb/swigpyvnlb.py")
    shutil.copyfile(swigpyvnlb_generic_lib, f"pyvnlb/_swigpyvnlb{ext}")

if found_swigpyvnlb_avx2:
    print(f"Copying {swigpyvnlb_avx2_lib}")
    shutil.copyfile("swigpyvnlb_avx2.py", "pyvnlb/swigpyvnlb_avx2.py")
    shutil.copyfile(swigpyvnlb_avx2_lib, f"pyvnlb/_swigpyvnlb_avx2{ext}")

long_description="""Pyvnlb is a library for video denising."""
setup(
    name='pyvnlb',
    version='1.0.0',
    description='A library for video image denoising.',
    long_description=long_description,
    url='https://github.com/gauenk/pyvnlb',
    author='Kent Gauen (copied from FAISS)',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='video non-local bayes',
    install_requires=['numpy'],
    packages=find_packages(include=['pyvnlb','pyvnlb.pylib*']),
    package_data={
        'pyvnlb': ['*.so', '*.pyd'],
    },

)
