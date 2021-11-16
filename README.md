PyVNLB 
=========================================
A Python API for Video Non-local Bayesian Denoising


Install
-------

```
$ git clone https://github.com/gauenk/pyvnlb/
$ cd pyvnlb
$ ./install.sh
```

Usage
-----

We expect the noisy input image to be shaped `(channels,nframes,height,width)` with
pixel values in range `[0,...,255.]`. The color channels are ordered RGB. Common noise levels include 10, 20, 50, etc.

```python

import vnlb.pylib as pyvnlb

# -- get data --
clean,noisy,std = pyvnlb.get_example_burst() # [0,...,255.]
print(noisy.shape) # (channels,nframes,height,width)

# -- exec TV-L1 Optical Flow --
fflow,bflow = pyvnlb.runPyFlow(noisy,std)

# -- exec Video Non-Local Bayes --
result = pyvnlb.runPyVnlb(noisy,std,{'fflow':fflow,'bflow':bflow})
denoised = result['denoised']

# -- compute denoising quality --
psnrs = pyvnlb.psnr(clean,denoised)
print("PSNRs:")
print(psnrs)

```


Dependencies
--------

The code depends on the following packages:
* [CBLAS](http://www.netlib.org/blas/#_cblas),
[LAPACKE](https://www.netlib.org/lapack/lapacke.html): operations with matrices
* OpenMP: parallelization [optional, but recommended]
* libpng, libtiff and libjpeg: image i/o

NOTE: By default, the code is compiled with OpenMP multithreaded
parallelization enabled (if your system supports it). Use the
`OMP_NUM_THREADS` enviroment variable to control the number of threads
used.

Credits
--------

This code provides is a Python wrapper over an implementation of the video denoising method (VNLB-H) described in:

[P. Arias, J.-M. Morel. "Video denoising via empirical Bayesian estimation of
space-time patches", Journal of Mathematical Imaging and Vision, 60(1),
January 2018.](https://link.springer.com/article/10.1007%2Fs10851-017-0742-4)


Please cite the publication if you use results obtained with this code in your research. 

* Original Code [linked here](https://github.com/pariasm/vnlb)
* C++ (and Primary) Author: Pablo Arias <pariasm@gmail.com>
* For computing the optical flow, it includes [the IPOL
implementation](http://www.ipol.im/pub/art/2013/26/) of
the [TV-L1 optical flow method of Zack and Pock and
Bischof](https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22).
* For image I/O, we use [Enric Meinhardt's iio](https://github.com/mnhrdt/iio).
* For SWIG-Python, Kent Gauen wrote this wrapper <kent.gauen@gmail.com>


LICENSE
-------

Licensed under the GNU Affero General Public License v3.0, see `LICENSE`.
