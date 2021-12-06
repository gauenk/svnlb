
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

#include <vnlb/cpp/utils/VnlbAsserts.h>
#include <vnlb/cpp/vnlb/VideoNLBayes.hpp>

#include <vnlb/cpp/pybind/interface.h>
#include <vnlb/cpp/pybind/vnlb/interface.h>

extern "C" {
#include <vnlb/cpp/flow/tvl1flow_lib.h>
#include <vnlb/cpp/video_io/iio.h>
}

void runFlatAreasCpp(FlatAreaParams& flat_params,VideoNLB::nlbParams& params) {

  /*****************

   initialize tensors

  *****************/
  if (params.verbose){
    VideoNLB::printNlbParameters(params);
  }

  // unpack sim params
  int chnls = flat_params.chnls;
  int nSimP = flat_params.nSimP;
  int gsize = flat_params.gsize;
  bool& flatPatch = flat_params.flatPatch;
  bool& flatAreas = flat_params.flatAreas;
  Video<float> groupNoisy,groupBasic;

  // setup video data
  std::memcpy(groupNoisy.data.data(),flat_params.groupNoisy,gsize);
  std::memcpy(groupBasic.data.data(),flat_params.groupBasic,gsize);

  // init
  // flatPatch = computeFlatArea(groupNoisy, groupBasic, params, nSimP, chnls);

  // write it back
  std::memcpy(flat_params.groupNoisy,groupNoisy.data.data(),gsize);
  std::memcpy(flat_params.groupBasic,groupBasic.data.data(),gsize);

}
