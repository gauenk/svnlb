#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <vnlb/cpp/flow/defaults.h>
#include <vnlb/cpp/vnlb/VideoNLBayes.hpp>

#include <vnlb/cpp/pybind/interface.h>
#include <vnlb/cpp/pybind/vnlb/interface.h>


using namespace std;

/// (args,tensors,step) -> (params)
void setVnlbParamsCpp(VideoNLB::nlbParams& params, const VnlbTensors& tensors, int step){

  // init
  float sigma = params.sigma;
  int search_space = params.sizeSearchWindow;
  int num_patches = params.nSimilarPatches;
  int rank = params.rank;
  float thresh = params.variThres;
  float beta = params.beta;
  bool flat_areas = params.flatAreas;
  bool couple_ch = params.coupleChannels;
  bool aggreBoost = params.aggreBoost;
  int patch_step = params.procStep;

  // bools to set; we don't have "inputs" like in c++
  bool set_sizePatch = params.set_sizePatch;
  bool set_sizePatchTime = params.set_sizePatchTime;
  bool set_nSim = params.set_nSim;
  bool set_rank = params.set_rank;
  bool set_aggreBoost = params.set_aggreBoost;
  bool set_procStep = params.set_procStep;

  // vars to set default
  // int ps_x = params.sizeSearchWindow;
  // int ps_t = params.sizeSearchTimeFwd;
  int ps_x = params.sizePatch;
  int ps_t = params.sizePatchTime;
  ps_x = (set_sizePatch) ? ps_x : -1;
  ps_t = (set_sizePatchTime) ? ps_t : -1;
  VideoSize img_sz(tensors.w,tensors.h,tensors.t,tensors.c);

  // set default
  VideoNLB::defaultParameters(params, ps_x, ps_t, step, sigma, img_sz, params.verbose);

  // Override with command line parameters
  if (set_sizePatch) VideoNLB::setSizeSearchWindow(params, (unsigned)search_space);
  if (set_nSim) VideoNLB::setNSimilarPatches(params, (unsigned)num_patches);
  if (set_aggreBoost) params.aggreBoost = aggreBoost;
  if (set_procStep) params.procStep = patch_step;
  if (set_rank) params.rank = rank;
  if (thresh        >= 0) params.variThres = thresh;
  if (beta         >= 0) params.beta = beta;
  params.flatAreas = flat_areas;
  params.coupleChannels = couple_ch;

  if (params.verbose){
    VideoNLB::printNlbParameters(params);
  }
}

