
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <vnlb/cpp/src/pybind/py_res.h>
#include <vnlb/cpp/src/VNLBayes/VideoNLBayes.hpp>

using namespace std;


void setVnlbParams(const PyVnlbParams& args, VideoNLB::nlbParams params, int step){

  // init 
  int index = step-1;
  float sigma = *(args.sigma+index);
  VideoSize tmp;

  // set default 
  tmp.channels = args.print_params;
  VideoNLB::defaultParameters(params, -1, 0, step, sigma, tmp, false);

  // set from args 
  VideoNLB::setSizeSearchWindow(params, args.search_space[index]);
  VideoNLB::setNSimilarPatches(params, (unsigned)args.num_patches[index]);
  // params.rank = args.rank[index];
  // params.variThresh = args.thres[index];
  // params.beta = args.beta[index];
  // params.flatAreas = args.flat_area[index];
  // params.coupleChannels = args.couple_ch[index];
  // params.aggreBoost = args.aggeBoost[index];
  // params.procStep = args.patch_step[index];
  // params.sigmaBasic = args.sigmab[index];
  if (args.print_params){
    VideoNLB::printNlbParameters(params);
  }
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
  oracle.loadVideoFromPtr(args.oracle,w,h,c,t);
  noisy.loadVideoFromPtr(args.noisy,w,h,c,t);
  basic.loadVideoFromPtr(args.basic,w,h,c,t);
  if (args.use_flow){
    fflow.loadVideoFromPtr(args.fflow,w,h,c,t);
    bflow.loadVideoFromPtr(args.fflow,w,h,c,t); // yes, fflow
  }

  // update params 
  VideoNLB::nlbParams params1, params2;
  setVnlbParams(args,params1,1);
  setVnlbParams(args,params2,2);

  // Percentage or processed groups of patches over total number of pixels
  std::vector<float> groupsRatio;

  // Run denoising algorithm
  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  // copy back to arrays
  final.saveVideoToPtr(const_cast<float*>(args.final));

}
