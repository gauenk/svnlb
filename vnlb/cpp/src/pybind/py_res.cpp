
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


void setVnlbParams(const PyVnlbParams& args, VideoNLB::nlbParams& params, int step){

  // init 
  int index = step-1;
  float sigma = *(args.sigma+index);

  // set default 
  // int psX = args.ps;
  // int psT = 1;
  // VideoSize tmp;
  // tmp.frames = args.t;
  // tmp.channels = args.print_params;
  // VideoNLB::defaultParameters(prms1, patch_sizex1, patch_sizet1, 1, sigma, tmp, false);
  VideoSize img_sz(args.w,args.h,args.t,args.c);
  VideoNLB::defaultParameters(params, -1, -1, step, sigma, img_sz, args.verbose);

  // set from args 
  // VideoNLB::setSizeSearchWindow(params, args.search_space[index]);
  // VideoNLB::setNSimilarPatches(params, (unsigned)args.num_patches[index]);
  // params.rank = args.rank[index];
  // params.variThresh = args.thres[index];
  // params.beta = args.beta[index];
  // params.flatAreas = args.flat_area[index];
  // params.coupleChannels = args.couple_ch[index];
  // params.aggreBoost = args.aggeBoost[index];
  // params.procStep = args.patch_step[index];
  // params.sigmaBasic = args.sigmab[index];
  if (args.verbose){
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
  // std::fprintf(stdout,"(w,h,c,t): (%d,%d,%d,%d)\n",w,h,c,t);

  // load video from ptr
  oracle.loadVideoFromPtr(args.oracle,w,h,c,t);
  noisy.loadVideoFromPtr(args.noisy,w,h,c,t);
  basic.loadVideoFromPtr(args.basic,w,h,c,t);
  if (args.use_flow){
    fflow.loadVideoFromPtr(args.fflow,w,h,2,t);
    bflow.loadVideoFromPtr(args.fflow,w,h,2,t); // yes, fflow
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
  // std::fprintf(stdout,"final.sz: %d,%d,%d,%d\n",
  // 	       final.sz.width,final.sz.height,
  // 	       final.sz.channels,final.sz.frames);

  // copy back to arrays
  final.saveVideoToPtr(const_cast<float*>(args.final));
  final.saveVideo("deno_%03d.png", 0, 1);
  noisy.saveVideo("noisy_%03d.png", 0, 1);


}
