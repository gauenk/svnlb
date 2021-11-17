#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <vnlb/cpp/src/pybind/py_res.h>
#include <vnlb/cpp/lib/tvl1flow/defaults.h>
#include <vnlb/cpp/src/VNLBayes/VideoNLBayes.hpp>

using namespace std;


void setVnlbParams(const PyVnlbParams& args, VideoNLB::nlbParams& params, int step){

  // init 
  int index = step-1;
  float sigma = *(args.sigma+index);

  // set default 
  // int psX = args.ps;
  // int psT = 1;
  VideoSize img_sz(args.w,args.h,args.t,args.c);
  VideoNLB::defaultParameters(params, -1, -1, step, sigma, img_sz, args.verbose);

  // set from args 
  params.verbose = args.verbose;
  params.coupleChannels = false;
  // VideoNLB::setSizeSearchWindow(params, args.search_space[index]);
  // VideoNLB::setNSimilarPatches(params, (unsigned)args.num_patches[index]);
  // params.rank = args.rank[index];
  // params.variThresh = args.thres[index];
  // params.beta = args.beta[index];
  // params.flatAreas = args.flat_area[index];
  // params.coupleChannels = args.couple_ch[index];
  // params.aggreBoost = false;
  // params.procStep = args.patch_step[index];
  // params.sigmaBasic = args.sigmab[index];
  if (args.verbose){
    VideoNLB::printNlbParameters(params);
  }
}

void setTvFlowParams(const PyTvFlowParams& args, tvFlowParams& params){

  //read the parameters
  params.nproc = (args.nproc < 0) ? PAR_DEFAULT_NPROC : args.nproc;
  params.tau = (args.tau < 0) ? PAR_DEFAULT_TAU : args.tau;
  params.lambda = (args.plambda < 0) ? PAR_DEFAULT_LAMBDA : args.plambda;
  params.theta = (args.theta < 0) ? PAR_DEFAULT_THETA : args.theta;
  params.nscales = (args.nscales < 0) ? PAR_DEFAULT_NSCALES : args.nscales;
  params.fscale = (args.fscale < 0) ? PAR_DEFAULT_FSCALE : args.fscale;
  params.zfactor = (args.zfactor < 0) ? PAR_DEFAULT_ZFACTOR : args.zfactor;
  params.nwarps = (args.nwarps < 0) ? PAR_DEFAULT_NWARPS : args.nwarps;
  params.epsilon = (args.epsilon < 0) ? PAR_DEFAULT_EPSILON : args.epsilon;
  params.verbose = (args.verbose < 0) ? PAR_DEFAULT_VERBOSE : args.verbose;

  // nproc = 0;
  // tau = 0.25;
  // lambda = 0.2;
  // theta = 0.3;
  // nscales = 100;
  // fscale = 1;
  // zfactor = 0.5;
  // nwarps = 5;
  // epsilon = 0.01;
  // verbose = 0;

  //check parameters
  if (params.nproc < 0) {
    params.nproc = PAR_DEFAULT_NPROC;
    if (params.verbose) fprintf(stderr, "warning: "
			 "nproc changed to %d\n", params.nproc);
  }
  if (params.tau <= 0 || params.tau > 0.25) {
    params.tau = PAR_DEFAULT_TAU;
    if (params.verbose) fprintf(stderr, "warning: "
			 "tau changed to %g\n", params.tau);
  }
  if (params.lambda <= 0) {
    params.lambda = PAR_DEFAULT_LAMBDA;
    if (params.verbose) fprintf(stderr, "warning: "
			 "lambda changed to %g\n", params.lambda);
  }
  if (params.theta <= 0) {
    params.theta = PAR_DEFAULT_THETA;
    if (params.verbose) fprintf(stderr, "warning: "
			 "theta changed to %g\n", params.theta);
  }
  if (params.nscales <= 0) {
    params.nscales = PAR_DEFAULT_NSCALES;
    if (params.verbose) fprintf(stderr, "warning: "
			 "nscales changed to %d\n", params.nscales);
  }
  if (params.zfactor <= 0 || params.zfactor >= 1) {
    params.zfactor = PAR_DEFAULT_ZFACTOR;
    if (params.verbose) fprintf(stderr, "warning: "
			 "zfactor changed to %g\n", params.zfactor);
  }
  if (params.nwarps <= 0) {
    params.nwarps = PAR_DEFAULT_NWARPS;
    if (params.verbose) fprintf(stderr, "warning: "
			 "nwarps changed to %d\n", params.nwarps);
  }
  if (params.epsilon <= 0) {
    params.epsilon = PAR_DEFAULT_EPSILON;
    if (params.verbose) fprintf(stderr, "warning: "
			 "epsilon changed to %f\n", params.epsilon);
  }
}

