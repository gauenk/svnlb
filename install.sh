#!/bin/bash

cmake -B build
make -C build -j swigvnlb
cd ./build/vnlb/python
pip install . --user
cd ../../../
