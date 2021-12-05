
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


void runAggregation(VideoNLB::nlbParams& params, PyAggParams& agg_params,
                    int& nmasked){

  /*****************

   initialize params

  *****************/
  if (params.verbose){
    VideoNLB::printNlbParameters(params);
  }

  // unpack sim params
  unsigned& nSimP = agg_params.nSimP;

  // unpack shape
  int w = agg_params.w;
  int h = agg_params.h;
  int c = agg_params.c;
  int t = agg_params.t;
  unsigned channels = c;
  int img_sz = w*h*c*t;
  int thw = t*h*w;

  //
  // initialization
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
  const int groupSize = patch_num * patch_dim * patch_chnls;

  // images
  Video<float> imDeno, weights;
  Video<char> mask;
  imDeno.loadVideoFromPtr(agg_params.imDeno,w,h,c,t);
  weights.loadVideoFromPtr(agg_params.weights,w,h,1,t);
  mask.loadVideoFromPtr((char*)agg_params.mask,w,h,1,t);

  // groups
  float* f_ptr = agg_params.group;
  std::vector<float> group(f_ptr,f_ptr+groupSize);
  unsigned* ui_ptr = agg_params.indices;
  std::vector<unsigned> indices(ui_ptr,ui_ptr+nSimP);

  //
  // Aggregate Similar Patches
  //
  nmasked += computeAggregation(imDeno,weights,mask,group,
                                indices,params,nSimP);
  //
  // Copy Local Mem *back* to Pybind Interface
  //

  f_ptr = agg_params.imDeno;
  std::memcpy(f_ptr,imDeno.data.data(),img_sz * sizeof(float));
  f_ptr = agg_params.weights;
  std::memcpy(f_ptr,weights.data.data(),thw * sizeof(float));
  char* c_ptr = (char*)agg_params.mask;
  std::memcpy(c_ptr,mask.data.data(),thw * sizeof(char));

}

