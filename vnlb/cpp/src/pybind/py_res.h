
#pragma once

/****

The insertion point for the Python API

****/


/// Arguments for video denoising
struct PyVnlbParams {
PyVnlbParams() :
	  // --> image details <--
          t(0),
	  h(0),
	  w(0),
	  c(0),
	  ps(0),
	  k(0),
	  use_clean(0),
	  use_flow(0),
	  fflow(nullptr),
	  oracle(nullptr),
	  noisy(nullptr),
	  basic(nullptr),
	  clean(nullptr),
	  final(nullptr), 
	  // --> vnlb tuning params <--
	  search_space(nullptr),
	  num_patches(nullptr),
	  rank(nullptr),
	  thresh(nullptr),
	  beta(nullptr),
	  flat_areas(nullptr),
	  couple_ch(nullptr),
	  aggeBoost(nullptr),
	  patch_step(nullptr),
	  sigmaBasic(nullptr),
	  sigma(nullptr){}
	  

    /***

    --->   Image Info   <---

    ***/

    int t; // nframes
    int h; // height
    int w; // width
    int c; // color
    int ps; // patchsize radius on one direction
    int k; // the num of neighbors
    bool use_clean;
    bool use_flow;

    // noisy image to denoise
    const float* oracle;
    const float* noisy;
    const float* basic;
    const float* clean;
    const float* final;
    const float* fflow;

    /***

    --->  VNLB Tuning Params  <---

    ***/

    float* sigma;
    float* sigmaBasic;
    unsigned* search_space;
    unsigned* num_patches;
    unsigned* rank;
    float* thresh;
    float* beta;
    unsigned* patch_step;
    bool* flat_areas;
    bool* couple_ch;
    bool* aggeBoost;
    bool verbose;
    unsigned print_params;

};


/// Arguments for video denoising
struct PyTvFlowParams {
PyTvFlowParams() :
  // --> image details <--
  t(0),h(0),w(0),c(0),
    image1(nullptr),
    image2(nullptr),
    flow(nullptr),
  // --> vnlb tuning params <--
    nproc(-1),
    tau(-1),
    lambda(-1),
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

  const float* image1;
  const float* image2;
  const float* flow;
  
  /***

      --->  VNLB Tuning Params  <---

  ***/

  int   nproc;
  float tau;
  float lambda;
  float theta;
  int   nscales;
  int   fscale;
  float zfactor;
  int   nwarps;
  float epsilon;
  int   verbose;
  bool  testing;

};

void runVnlb(const PyVnlbParams& args);
void runTV1Flow(const PyTvFlowParams& args);


