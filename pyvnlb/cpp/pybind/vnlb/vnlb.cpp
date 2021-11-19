
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

#include <pyvnlb/cpp/pybind/vnlb/interface.h>
#include <pyvnlb/cpp/pybind/vnlb/parser.h>

extern "C" {
#include <pyvnlb/cpp/flow/tvl1flow_lib.h>
#include <pyvnlb/cpp/video_io/iio.h>
}


void runVnlb(const PyVnlbParams& args) {

  // Declarations
  Video<float> oracle, noisy, basic, final;
  Video<float> fflow, bflow;

  // unpack shape 
  int w = args.w;
  int h = args.h;
  int c = args.c;
  int t = args.t;

  // load video from ptr
  // oracle.loadVideoFromPtr(args.oracle,w,h,c,t); // not implemented
  noisy.loadVideoFromPtr(args.noisy,w,h,c,t);
  basic.loadVideoFromPtr(args.basic,w,h,c,t);
  if (args.use_flow){
    fflow.loadVideoFromPtr(args.fflow,w,h,2,t);
    bflow.loadVideoFromPtr(args.bflow,w,h,2,t);
  }

  // update params 
  VideoNLB::nlbParams params1, params2;
  setVnlbParams(args,params1,1);
  setVnlbParams(args,params2,2);

  // Percentage or processed groups of patches over total number of pixels
  std::vector<float> groupsRatio;

  // -- 1st step --
  auto tmp = params2.sizePatch;
  params2.sizePatch = 0;
  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  if (args.testing){
    basic.saveVideoToPtr(args.basic);
  }
  if (args.verbose)
    printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
		       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);

  // -- 2nd step --
  params1.sizePatch = 0;
  params2.sizePatch = tmp;
  // re-load noisy image, see git issue #9 in pariasm/vnlb
  noisy.loadVideoFromPtr(args.noisy,w,h,c,t); 
  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  if (args.verbose)
    printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
		       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);

  // copy back to arrays
  final.saveVideoToPtr(args.final);
}