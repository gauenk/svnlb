
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <chrono>
#include <ctime>

#include <string>
#include <sstream>
#include <float.h>

#include <pyvnlb/cpp/utils/VnlbAsserts.h>
#include <pyvnlb/cpp/vnlb/VideoNLBayes.hpp>

#include <pyvnlb/cpp/pybind/interface.h>
#include <pyvnlb/cpp/pybind/vnlb/interface.h>

extern "C" {
#include <pyvnlb/cpp/flow/tvl1flow_lib.h>
#include <pyvnlb/cpp/video_io/iio.h>
}


void runBayesEstimate(VideoNLB::nlbParams& params,
                      PyBayesEstimateParams& bayes_params){

  /*****************

   initialize params

  *****************/
  if (params.verbose){
    VideoNLB::printNlbParameters(params);
  }

  // unpack sim params
  unsigned& nSimP = bayes_params.nSimP;

  // unpack shape
  int w = bayes_params.w;
  int h = bayes_params.h;
  int c = bayes_params.c;
  int t = bayes_params.t;
  unsigned channels = c;

  //
  // initialization for output vars
  //

  // unpack variables used for allocation
  const bool step1 = params.isFirstStep;
  const unsigned sWx = params.sizeSearchWindow;
  const unsigned sWt_f = params.sizeSearchTimeFwd;
  const unsigned sWt_b = params.sizeSearchTimeBwd;
  const unsigned sWt = sWt_f + sWt_b + 1;// VIDEO
  const unsigned sPx = params.sizePatch;
  const unsigned sPt = params.sizePatchTime;
  const unsigned patch_num = sWx * sWx * sWt;
  const unsigned patch_dim = sPx * sPx * sPt * (params.coupleChannels ? channels : 1);
  const unsigned patch_chnls = params.coupleChannels ? 1 : channels;

  // initialize vectors
  int groupSize = patch_num * patch_dim * patch_chnls;

  // init vectors of the groups
  std::fprintf(stdout,"groupSize: %d\n",groupSize);
  float* ptr = bayes_params.groupNoisy;
  std::vector<float> groupNoisy(ptr,ptr+groupSize);
  ptr = bayes_params.groupBasic;
  std::vector<float> groupBasic(ptr,ptr+groupSize);

  // init matrix workspace vars
  VideoNLB::matWorkspace mat_ws;
  mat_ws.group     .resize(patch_num * patch_dim);
  mat_ws.covMat    .resize(patch_dim * patch_dim);
  mat_ws.center.resize(patch_dim * patch_chnls);

  //
  // Bayes Denoising!
  //

  bayes_params.rank_var += computeBayesEstimate(groupNoisy,groupBasic,
                                                mat_ws,params,nSimP,
                                                channels,false);
  //
  // Copy Local Mem *back* to Pybind Interface
  //

  ptr = bayes_params.mat_group;
  std::memcpy(ptr,mat_ws.group.data(),patch_num * patch_dim * sizeof(float));
  ptr = bayes_params.mat_covMat;
  std::memcpy(ptr,mat_ws.covMat.data(),patch_dim * patch_dim * sizeof(float));
  ptr = bayes_params.mat_center;
  std::memcpy(ptr,mat_ws.center.data(),patch_dim * patch_chnls * sizeof(float));
  ptr = bayes_params.mat_covEigVecs;
  std::memcpy(ptr,mat_ws.covEigVecs.data(),patch_dim * params.rank * sizeof(float));
  ptr = bayes_params.mat_covEigVals;
  std::memcpy(ptr,mat_ws.covEigVals.data(),patch_dim * sizeof(float));

  ptr = bayes_params.groupNoisy;
  std::memcpy(ptr,groupNoisy.data(), groupSize * sizeof(float));
  ptr = bayes_params.groupBasic;
  std::memcpy(ptr,groupBasic.data(), groupSize * sizeof(float));

}

