

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
