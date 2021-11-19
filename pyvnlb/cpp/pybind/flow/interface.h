

#pragma once

/// Arguments for video denoising
struct PyTvFlowParams {
PyTvFlowParams() :
  // --> image details <--
  t(0),h(0),w(0),c(0),
    burst(nullptr),
    fflow(nullptr),
    bflow(nullptr),
  // --> vnlb tuning params <--
    direction(-1),
    nproc(-1),
    tau(-1),
    plambda(-1),
    nscales(-1),
    fscale(-1),
    zfactor(-1),
    nwarps(-1),
    epsilon(-1),
    verbose(0),
    testing(0) {}

  /***

      --->   Image Info   <---

  ***/

  int t; // nframes
  int h; // height
  int w; // width
  int c; // color

  // image for flow

  float* burst;
  float* fflow;
  float* bflow;
  
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

void runTV1Flow(const PyTvFlowParams& args);
