Python Bindings for VNLB | Non-local Bayesian Video Denoising
=========================================

* C++ (and Primary) Author    : Pablo Arias <pariasm@gmail.com>, see `AUTHORS`
* Copyright : (C) 2019, Pablo Arias <pariasm@gmail.com>
* Python Bindings Author: Kent Gauen

OVERVIEW
--------

This code provides is a Python wrapper over an implementation of the video denoising method VNLB-H described in:

[P. Arias, J.-M. Morel. "Video denoising via empirical Bayesian estimation of
space-time patches", Journal of Mathematical Imaging and Vision, 60(1),
January 2018.](https://link.springer.com/article/10.1007%2Fs10851-017-0742-4)

Please cite the publication if you use results obtained with this code in your
research.

The following libraries are also included as part of the code:
* For computing the optical flow, it includes [the IPOL
implementation](http://www.ipol.im/pub/art/2013/26/) of
the [TV-L1 optical flow method of Zack and Pock and
Bischof](https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22).
* For image I/O, we use [Enric Meinhardt's iio](https://github.com/mnhrdt/iio).

The original git repo is [available here](https://github.com/pariasm/vnlb/).

**Dependencies:** The code depends on the following packages:
* [CBLAS](http://www.netlib.org/blas/#_cblas),
[LAPACKE](https://www.netlib.org/lapack/lapacke.html): operations with matrices
* OpenMP: parallelization [optional, but recommended]
* libpng, libtiff and libjpeg: image i/o

**Compilation:** 
Compilation was tested on Ubuntu Linux 16.04 and 18.04.
Configure and compile the source code using cmake and make.
It is recommended that you create a folder for building:
```
$ mkdir build; cd build
$ cmake ..
$ make
```

Binaries will be created in `build/bin folder`.

NOTE: By default, the code is compiled with OpenMP multithreaded
parallelization enabled (if your system supports it). Use the
`OMP_NUM_THREADS` enviroment variable to control the number of threads
used.

USAGE
-----

Please [see original github repo](https://github.com/pariasm/vnlb/) for usage of original code. The python binding usage is coming soon...


LICENSE
-------

Licensed under the GNU Affero General Public License v3.0, see `LICENSE`.
