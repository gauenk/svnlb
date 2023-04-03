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
# shutil.rmtree("vnlb", ignore_errors=True)
shutil.rmtree("svnlb", ignore_errors=True)
os.mkdir("svnlb")
shutil.copytree("swig", "svnlb/swig")
shutil.copytree("testing", "svnlb/testing")
shutil.copytree("utils", "svnlb/utils")
shutil.copytree("cpu", "svnlb/cpu")
# shutil.copytree("gpu", "svnlb/gpu")
shutil.copytree("exps", "svnlb/exps")
shutil.copyfile("__init__.py", "svnlb/__init__.py")
shutil.copyfile("loader.py", "svnlb/loader.py")

ext = ".pyd" if platform.system() == 'Windows' else ".so"
prefix = "Release/" * (platform.system() == 'Windows')

swigvnlb_generic_lib = f"{prefix}_swigvnlb{ext}"
swigvnlb_avx2_lib = f"{prefix}_swigvnlb_avx2{ext}"

found_swigvnlb_generic = os.path.exists(swigvnlb_generic_lib)
found_swigvnlb_avx2 = os.path.exists(swigvnlb_avx2_lib)

assert (found_swigvnlb_generic or found_swigvnlb_avx2), \
    f"Could not find {swigvnlb_generic_lib} or " \
    f"{swigvnlb_avx2_lib}. Vnlb may not be compiled yet."

if found_swigvnlb_generic:
    print(f"Copying {swigvnlb_generic_lib}")
    shutil.copyfile("swigvnlb.py", "svnlb/swigvnlb.py")
    shutil.copyfile(swigvnlb_generic_lib, f"svnlb/_swigvnlb{ext}")

if found_swigvnlb_avx2:
    print(f"Copying {swigvnlb_avx2_lib}")
    shutil.copyfile("swigvnlb_avx2.py", "svnlb/swigvnlb_avx2.py")
    shutil.copyfile(swigvnlb_avx2_lib, f"svnlb/_swigvnlb_avx2{ext}")

long_description="""SVNLB (Python-Swig Vnlb) is a library for video denoising."""
setup(
    name='svnlb',
    version='1.0.0',
    description='A library for video image denoising.',
    long_description=long_description,
    url='https://github.com/gauenk/svnlb',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='video non-local bayes',
    install_requires=['numpy'],
    packages=find_packages(include=['svnlb',
                                    'svnlb.swig*',
                                    'svnlb.cpu*',
                                    'svnlb.gpu*',
                                    'svnlb.utils*',
                                    'svnlb.testing*',
                                    'svnlb.exps*'
    ]),
    package_data={
        'svnlb': ['*.so', '*.pyd'],
    },

)
