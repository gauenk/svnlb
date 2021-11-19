
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <pyvnlb/cpp/pybind/py_res.h>
#include <pyvnlb/cpp/flow/defaults.h>
#include <pyvnlb/cpp/vnlb/VideoNLBayes.hpp>

using namespace std;

void setVnlbParams(const PyVnlbParams& args, VideoNLB::nlbParams& params, int step);
void setTvFlowParams(const PyTvFlowParams& args, tvFlowParams& params);
