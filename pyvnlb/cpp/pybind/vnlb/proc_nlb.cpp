
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


void processNLBayesCpp(VideoNLB::nlbParams& params,
                        const VnlbTensors& tensors,
                       int& group_counter,
                       int border) {

  // -- Declarations --
  Video<float> oracle, noisy, basic, imFinal;
  Video<float> fflow, bflow;

  // -- unpack shape --
  int w = tensors.w;
  int h = tensors.h;
  int c = tensors.c;
  int t = tensors.t;

  // -- load video from ptr --
  noisy.loadVideoFromPtr(tensors.noisy,w,h,c,t);
  basic.loadVideoFromPtr(tensors.basic,w,h,c,t);
  if (tensors.use_flow){
    fflow.loadVideoFromPtr(tensors.fflow,w,h,2,t);
    bflow.loadVideoFromPtr(tensors.bflow,w,h,2,t);
  }
  if (tensors.use_oracle){
    oracle.loadVideoFromPtr(tensors.oracle,w,h,c,t);
  }

  // -- get crops --
  unsigned nParts = 1;
  std::vector<Video<float> > imEmpty(nParts);
  std::vector<VideoUtils::TilePosition > imCrops(nParts);
  VideoUtils::subDivideTight(noisy, imEmpty, imCrops, border, nParts);

  // RGB to YUV
  VideoUtils::transformColorSpace(noisy, true);
  // VideoUtils::transformColorSpace(clean, true);
  if (!params.isFirstStep) VideoUtils::transformColorSpace(basic, true);

  // -- process step --
  int ngroups = processNLBayes(noisy,fflow,bflow,basic,imFinal,
                               params,imCrops[0],oracle);
  group_counter += ngroups;

  // YUV to RGB
  VideoUtils::transformColorSpace(noisy, false);
  // VideoUtils::transformColorSpace(clean, false);
  VideoUtils::transformColorSpace(basic, false);


  // handle "falling action"
  basic.saveVideoToPtr(tensors.basic);
  // if (params.verbose)
  //   printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
  //   	       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);

  // copy back to arrays
  imFinal.saveVideoToPtr(tensors.denoised);

}
