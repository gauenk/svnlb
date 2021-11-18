PyVNLB 
=========================================
A Python API for Video Non-Local Bayesian Denoising ([C++ code originally from Pablo Arias](https://github.com/pariasm/vnlb))


Install
-------

```
$ git clone https://github.com/gauenk/pyvnlb/
$ cd pyvnlb
$ ./install.sh
$ python -m pip install -r requirements.txt --user
```

Usage
-----

We expect the images to be shaped `(nframes,channels,height,width)` with
pixel values in range `[0,...,255.]`. The color channels are ordered RGB. Common examples of noise levels are 10, 20 and 50. See [scripts/example.py](https://github.com/gauenk/pyvnlb/blob/master/scripts/example.py) for more details.

```python
import numpy as np
import vnlb.pylib as pyvnlb

# -- get data --
clean = 255.*np.random.rand(5,3,64,64)
# (nframes,channels,height,width)

# -- add noise --
std = 20.
noisy = np.random.normal(clean,scale=std)

# -- TV-L1 Optical Flow --
fflow,bflow = pyvnlb.runPyFlow(noisy,std)

# -- Video Non-Local Bayes --
result = pyvnlb.runPyVnlb(noisy,std,{'fflow':fflow,'bflow':bflow})
denoised = result['denoised']

# -- compute denoising quality --
psnrs = pyvnlb.psnr(clean,denoised)
print("PSNRs:")
print(psnrs)

```

Comparing with C++ Code
---

The outputs from the Python API and the C++ Code are exactly equal. To demonstrate this claim, we provide the `scripts/compare_cpp.py` script. We have two examples of the [C++ Code](https://github.com/pariasm/vnlb) output ready for download using the respective `scripts/download_davis*.sh` files. To run the data downloading scripts, type:

```
$ ./scripts/download_davis_64x64.sh
```

To run the comparison, type:

```
$ export OMP_NUM_THREADS=4
$ python scripts/compare_cpp.py
```

The script prints the below table. Each element of the table is the sum of the absolute relative error between the outputs from the Python API and C++ Code.

|                   |   noisyForFlow |   noisyForVnlb |   fflow |   bflow |   basic |   denoised |
|:------------------|---------------:|---------------:|--------:|--------:|--------:|-----------:|
| Total Error (cv2) |    0.000505755 |              0 | 504.308 |  21.643 |       0 |          0 |
| Total Error (cpp) |    0           |              0 |   0     |   0     |       0 |          0 |


Details can be found in [COMPARE.md](https://github.com/gauenk/pyvnlb/blob/master/COMPARE.md)

Dependencies
--------

The code depends on the following packages:
* [CBLAS](http://www.netlib.org/blas/#_cblas), [LAPACKE](https://www.netlib.org/lapack/lapacke.html): operations with matrices
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
