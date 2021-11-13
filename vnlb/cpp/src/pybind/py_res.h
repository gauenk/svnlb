

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
	  noisy(nullptr),
	  clean(nullptr),
	  final(nullptr), 
	  // --> vnlb tuning params <--
	  search_space(nullptr),
	  num_patches(nullptr),
	  rank(nullptr),
	  thres(nullptr),
	  beta(nullptr),
	  flat_area(nullptr),
	  couple_ch(nullptr),
	  aggeBoost(nullptr),
	  patch_step(nullptr),
	  sigmab(nullptr),
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
    const float* noisy;
    const float* clean;
    const float* final;
    const float* fflow;

    /***

    --->  VNLB Tuning Params  <---

    ***/

    float[2] sigma;
    float[2] sigmaBasic;
    unsigned[2] search_space;
    unsigned[2] num_patches;
    unsigned[2] rank;
    float[2] thresh;
    float[2] beta;
    unsigned[2] patch_step;
    bool[2] flat_areas;
    bool[2] couple_ch;
    bool[2] aggeBoost;
    bool verbose;
    bool print_params;

};

void runVnlb(const PyVnlbParams& args);


