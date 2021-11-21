
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

void runVnlbTimed(const PyVnlbParams& args, const VnlbTensors& tensors) {

  // Declarations
  Video<float> oracle, noisy, basic, final;
  Video<float> fflow, bflow;

  // unpack shape
  int w = tensors.w;
  int h = tensors.h;
  int c = tensors.c;
  int t = tensors.t;

  // load video from ptr
  noisy.loadVideoFromPtr(tensors.noisy,w,h,c,t);
  basic.loadVideoFromPtr(tensors.basic,w,h,c,t);
  if (tensors.use_flow){
    fflow.loadVideoFromPtr(tensors.fflow,w,h,2,t);
    bflow.loadVideoFromPtr(tensors.bflow,w,h,2,t);
  }
  if (tensors.use_oracle){
    oracle.loadVideoFromPtr(tensors.oracle,w,h,c,t);
  }

  // update params
  VideoNLB::nlbParams params1, params2;
  setVnlbParams(args,tensors,params1,1);
  setVnlbParams(args,tensors,params2,2);

  // Percentage or processed groups of patches over total number of pixels
  std::vector<float> groupsRatio;

  // Run denoising algorithm
  auto tmp = params2.sizePatch;
  params2.sizePatch = 0;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
  	    << "elapsed time: " << elapsed_seconds.count() << "s\n";


  if (args.testing){
    basic.saveVideoToPtr(tensors.basic);
  }
  if (args.verbose)
    printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
		       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);

  params1.sizePatch = 0;
  params2.sizePatch = tmp;
  // re-load noisy image, see git issue #9 in pariasm/vnlb
  noisy.loadVideoFromPtr(tensors.noisy,w,h,c,t);
  start = std::chrono::system_clock::now();
  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
  	    << "elapsed time: " << elapsed_seconds.count() << "s\n";

  if (args.verbose)
    printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
		       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);

  // copy back to arrays
  final.saveVideoToPtr(tensors.denoised);

}

