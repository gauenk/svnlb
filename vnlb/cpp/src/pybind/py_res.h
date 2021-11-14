
#pragma once

/****

The insertion point for the Python API

****/


/// Arguments to brute-force GPU k-nearest neighbor searching
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

void runVnlb(const PyVnlbParams& args);


