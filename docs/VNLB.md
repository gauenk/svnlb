VNLB
---

This file describes how to create the output files to compate the C++ Code with this Python API. In this mini-tutorial, the Python API repo is called `pyvnlb` and the original C++ code is called `vnlb`.

First, the original C++ code must be installed separately [from this link](https://github.com/pariasm/vnlb). 

```
$ git clone https://github.com/pariasm/vnlb
$ mkdir build; cd build
$ cmake ..
$ make
```

Next we show how to use this code to create the ground-truth images. We first move to the directory with the executable files,

```
$ cd vnlb/build/bin/
```

Then we run the C++ Code. This particular execution uses frames from 0 to (and including) 4 for a total of 5 frames. The noise level is 20. 

```
$ export OMP_NUM_THREADS=4
$ ./vnlb-gt.sh $PYVNLB_HOME/data/davis_baseball_64x64/%05d.jpg 0 4 20 $PYVNLB_HOME/data/davis_baseball_64x64/vnlb/ "-px1 7 -pt1 2 -px2 7 -pt2 2 -verbose"
```

Finally, write the noise level to a text file in the output directory,

```
$ echo "20" > $PYVNLB_HOME/pyvnlb/data/davis_baseball_64x64/vnlb/sigma.txt
```
