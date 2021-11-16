
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <vnlb/cpp/src/pybind/py_res.h>
#include <vnlb/cpp/lib/tvl1flow/defaults.h>
#include <vnlb/cpp/src/VNLBayes/VideoNLBayes.hpp>

using namespace std;

void setVnlbParams(const PyVnlbParams& args, VideoNLB::nlbParams& params, int step);
void setTvFlowParams(const PyTvFlowParams& args, tvFlowParams& params);
