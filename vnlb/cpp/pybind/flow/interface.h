

#pragma once


#include <vnlb/cpp/flow/defaults.h>
#include <vnlb/cpp/pybind/interface.h>


// Arguments for video denoising
struct PyTvFlowParams {
PyTvFlowParams() :
  // --> image details <--
  direction(-1),
    nproc(-1),
    tau(-1),
    theta(-1),
    plambda(-1),
    nscales(-1),
    fscale(-1),
    zfactor(-1),
    nwarps(-1),
    epsilon(-1),
    verbose(0),
    testing(0) {}


  /***

  --->  VNLB Tuning Params  <---

  ***/

  int   direction;
  int   nproc;
  float tau;
  float plambda;
  float theta;
  int   nscales;
  int   fscale;
  float zfactor;
  int   nwarps;
  float epsilon;
  int   verbose;
  bool  testing;

};

void runTV1Flow(const PyTvFlowParams& args, const VnlbTensors& tensors);
void setTvFlowParams(const PyTvFlowParams& args, tvFlowParams& params);
