

#pragma once

#include <pyvnlb/cpp/vnlb/VideoNLBayes.hpp>
#include <pyvnlb/cpp/pybind/interface.h>

/// Arguments for video denoising
struct PyVnlbParams {
PyVnlbParams() :
  // --> image details <--
  ps_x(nullptr),
    ps_t(nullptr),
    num_patches(nullptr),
    sizeSearchWindow(nullptr),
    sizeSearchTimeFwd(nullptr),
    sizeSearchTimeBwd(nullptr),
    sigma(nullptr),
    tau(nullptr),
    rank(nullptr),
    var_mode(0),
    thresh(nullptr),
    beta(nullptr),
    flat_areas(nullptr),
    couple_ch(nullptr),
    aggreBoost(nullptr),
    procStep(nullptr),
    sigmaBasic(nullptr),
    use_default(0),
    testing(0),
    verbose(0){}

  // * (nlbParams - PyVnlbParams): gamma,variThres,isFirstStep,onlyFrame,
  // * (PyVnlbParams - nlbParams): thresh, use_default, testing,

  /***
      Sim Search Params
  ***/

  int* ps_x; // patchsize [spatial]
  int* ps_t; // patchsize [temporal]
  int* num_patches; // the num of neighbors for denoising
  int* sizeSearchWindow; // Spatial search diameter
  int* sizeSearchTimeFwd; // Number of search frames (forward)
  int* sizeSearchTimeBwd; // Number of search fraems (backward)
  float* tau; // patch distance threshold
  bool var_mode; // use H or S

  /***
      VNLB Tuning Params
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
  unsigned print_params;
  bool use_default;
  bool testing;
  bool verbose;

};


struct PySimSearchParams {
PySimSearchParams() :
  nParts(1),nSimP(0),pidx(0),all_pix(0){}

  unsigned nParts;
  unsigned nSimP;
  unsigned pidx;
  bool all_pix;

};


struct PyBayesEstimateParams {
PyBayesEstimateParams() :
  t(0),c(0),h(0),w(0),
  groupNoisy(nullptr),groupBasic(nullptr),
    mat_group(nullptr),mat_center(nullptr),
    mat_covMat(nullptr),mat_covEigVecs(nullptr),
    mat_covEigVals(nullptr),nSimP(0),rank_var(0) {}

  // --> to denoise <--
  float* groupNoisy;
  float* groupBasic;
  int t,c,h,w;

  // --> mat workspace <--
  float* mat_group;
  float* mat_center;
  float* mat_covMat;
  float* mat_covEigVecs;
  float* mat_covEigVals;

  // --> num of similar patches <--
  unsigned nSimP;

  // -> an output <-
  float rank_var;

};

struct PyAggParams {
PyAggParams() :
  t(0),c(0),h(0),w(0),
    imDeno(nullptr),weights(nullptr),mask(nullptr),
    group(nullptr),indices(nullptr), nSimP(0) {}

  // --> shapes <--
  int t,c,h,w;

  // --> to denoise <--
  float* imDeno;
  float* weights;
  void* mask;

  // --> patches <--
  float* group;
  unsigned* indices;

  // --> num of similar patches <--
  unsigned nSimP; // num of similar patches

};

struct MaskParams {
MaskParams() :
  mask(nullptr),
    nframes(0),width(0),height(0),
    origin_t(0),origin_h(0),origin_w(0),
    ending_t(0),ending_h(0),ending_w(0),
    step_t(1),step_h(1),step_w(1),
    ps(0),ps_t(0),sWx(0),sWt(0) {};

  // -- init --
  void* mask;

  int nframes;
  int width;
  int height;

  int origin_t;
  int origin_h;
  int origin_w;

  int ending_t;
  int ending_h;
  int ending_w;

  int step_t;
  int step_h;
  int step_w;

  int ps;
  int ps_t;
  int sWx;
  int sWt;

};

struct CovMatParams {
CovMatParams():
  pdim(0),rank(0),nSimP(0),gsize(0),
    groups(nullptr),covMat(nullptr),
    covEigVals(nullptr),covEigVecs(nullptr) {}

  // size of ints
  int pdim;
  int rank;
  int nSimP;
  int gsize;

  // arrays galore
  float* groups;
  float* covMat;
  float* covEigVals;
  float* covEigVecs;

};


/* void runVnlb(const PyVnlbParams& args, const VnlbTensors& tensors); */
/* void runVnlbTimed(const PyVnlbParams& args, const VnlbTensors& tensors); */
/* void setVnlbParamsCpp(const PyVnlbParams& args, const VnlbTensors& tensors, */
/*                       VideoNLB::nlbParams& params, int step); */
void runVnlb(VideoNLB::nlbParams& params1,
             VideoNLB::nlbParams& params2,
             const VnlbTensors& tensors);
/* void runVnlbTimed(VideoNLB::nlbParams& params1, */
/*              VideoNLB::nlbParams& params2, */
/*              const VnlbTensors& tensors); */
void setVnlbParamsCpp(VideoNLB::nlbParams& params, const VnlbTensors& tensors,int step);
void runSimSearch(VideoNLB::nlbParams& params,
                  const VnlbTensors& tensors,
                  PySimSearchParams& sim_params);
void runBayesEstimate(VideoNLB::nlbParams& params,
                      PyBayesEstimateParams& bayes_params);
void runAggregation(VideoNLB::nlbParams& params,
                    PyAggParams& agg_params,
                    int& nmasked);
void processNLBayesCpp(VideoNLB::nlbParams& params,
                       const VnlbTensors& tensors,
                       int& group_counter, int border);
void init_mask_cpp(MaskParams params, int& ngroups);
void computeCovMatCpp(CovMatParams params);

