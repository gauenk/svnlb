
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

/// Arguments for checking "loadvideo" and "readiio"
struct ReadVideoParams {
ReadVideoParams() :
  // --> image details <--
  t(0),h(0),w(0),c(0),
    read_video(nullptr),
    video_paths(""),
    first_frame(0),
    last_frame(0),
    frame_step(1),
    verbose(0) {}
  
  // -- image params --
  bool verbose;
  int t,h,w,c;
  int first_frame,last_frame,frame_step;
  float* read_video;
  const char* video_paths;
};


void runVnlb(const PyVnlbParams& args);
void runVnlbTimed(const PyVnlbParams& args);
void runTV1Flow(const PyTvFlowParams& args);
/* void testLoadVideo(const ReadVideoParams& args); */
/* void testIIORead(const ReadVideoParams& args); */
void readVideoForVnlb(const ReadVideoParams& args);
void readVideoForFlow(const ReadVideoParams& args);
