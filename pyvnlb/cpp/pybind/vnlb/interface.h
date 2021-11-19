

#pragma once 

/// Arguments for video denoising
struct PyVnlbParams {
PyVnlbParams() :
	  // --> image details <--
          t(0),
	  h(0),
	  w(0),
	  c(0),
	  ps_x(nullptr),
	  ps_t(nullptr),
	  num_patches(nullptr),
	  sizeSearchWindow(nullptr),
	  sizeSearchTimeFwd(nullptr),
	  sizeSearchTimeBwd(nullptr),
	  tau(nullptr),
	  use_clean(0),
	  use_flow(0),
	  fflow(nullptr),
	  bflow(nullptr),
	  oracle(nullptr),
	  noisy(nullptr),
	  basic(nullptr),
	  clean(nullptr),
	  final(nullptr), 
	  // --> vnlb tuning params <--
	  use_default(0),
	  rank(nullptr),
	  var_mode(0),
	  thresh(nullptr),
	  beta(nullptr),
	  flat_areas(nullptr),
	  couple_ch(nullptr),
	  aggreBoost(nullptr),
	  procStep(nullptr),
	  sigmaBasic(nullptr),
	  sigma(nullptr),
	  testing(0){}
	  

    /***

    --->   Image Info   <---

    ***/

    int t; // nframes
    int h; // height
    int w; // width
    int c; // color
    int* ps_x; // patchsize [spatial]
    int* ps_t; // patchsize [temporal]
    int* num_patches; // the num of neighbors for denoising
    int* sizeSearchWindow; // Spatial search diameter
    int* sizeSearchTimeFwd; // Number of search frames (forward)
    int* sizeSearchTimeBwd; // Number of search fraems (backward)
    float* tau; // patch distance threshold
    bool use_clean;
    bool use_flow;
    bool var_mode; // use H or S

    // noisy image to denoise
    float* oracle;
    float* noisy;
    float* basic;
    float* clean;
    float* final;
    float* fflow;
    float* bflow;

    /***

    --->  VNLB Tuning Params  <---

    ***/

    float* sigma;
    float* sigmaBasic;
    int* rank;
    float* thresh;
    float* beta;
    int* procStep;
    bool* flat_areas;
    bool* couple_ch;
    bool* aggreBoost;
    bool verbose;
    bool use_default;
    unsigned print_params;
    bool testing;

};


struct PySimSearchParams {
PySimSearchParams() :
    gNoisy(nullptr),
    gBasic(nullptr),
    indices(nullptr),
    pixel_index(nullptr),
    whole_image(0){}

  float* gNoisy;
  float* gBasic;
  unsigned* indices;
  unsigned* pixel_index;
  bool whole_image;

};

void runVnlb(const PyVnlbParams& args);
void runVnlbTimed(const PyVnlbParams& args);
