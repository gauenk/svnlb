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
shutil.rmtree("vnlb", ignore_errors=True)
os.mkdir("vnlb")
shutil.copytree("pylib", "vnlb/pylib")
shutil.copytree("benchmarks", "vnlb/benchmarks")
shutil.copyfile("__init__.py", "vnlb/__init__.py")
shutil.copyfile("loader.py", "vnlb/loader.py")

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
    shutil.copyfile("swigvnlb.py", "vnlb/swigvnlb.py")
    shutil.copyfile(swigvnlb_generic_lib, f"vnlb/_swigvnlb{ext}")

if found_swigvnlb_avx2:
    print(f"Copying {swigvnlb_avx2_lib}")
    shutil.copyfile("swigvnlb_avx2.py", "vnlb/swigvnlb_avx2.py")
    shutil.copyfile(swigvnlb_avx2_lib, f"vnlb/_swigvnlb_avx2{ext}")

long_description="""Vnlb is a library for video denising."""
setup(
    name='vnlb',
    version='1.0.0',
    description='A library for video image denoising.',
    long_description=long_description,
    url='https://github.com/gauenk/pyvnlb',
    author='Kent Gauen (copied from FAISS)',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='video non-local bayes',
    install_requires=['numpy'],
    packages=['vnlb', 'vnlb.pylib', 'vnlb.benchmarks'],
    package_data={
        'vnlb': ['*.so', '*.pyd'],
    },

)

# -- path to installed file --
def binaries_directory():
    """Return the installation directory, or None"""
    if '--user' in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        py_version = '%s.%s' % (sys.version_info[0], sys.version_info[1])
        paths = (s % (py_version) for s in (
            sys.prefix + '/lib/python%s/dist-packages/',
            sys.prefix + '/lib/python%s/site-packages/',
            sys.prefix + '/local/lib/python%s/dist-packages/',
            sys.prefix + '/local/lib/python%s/site-packages/',
            '/Library/Python/%s/site-packages/',
        ))

    for path in paths:
        if os.path.exists(path):
            return path
    print('no installation path found', file=sys.stderr)
    return None
fpath = binaries_directory()
gpath = glob.glob(os.path.join(fpath,"vnlb*"))
msg = f"Please remove old install at path [{fpath}] before finishing Python install"
print(msg)
# assert len(gpath) == 1, msg
print(gpath)
# fpath = gpath[0]
# run chmod 775 on the installed egg.
# os.chmod(fpath,
#          stat.S_IRUSR |
#          stat.S_IWUSR |
#          stat.S_IXUSR |
#          stat.S_IRGRP |
#          stat.S_IWGRP |
#          stat.S_IXGRP |
#          stat.S_IROTH |
#          stat.S_IXOTH )

