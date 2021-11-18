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
  float sigma = args.sigma[index];

  // set default 
  VideoSize img_sz(args.w,args.h,args.t,args.c);
  VideoNLB::defaultParameters(params, args.ps_x[index], args.ps_t[index],
			      step, sigma, img_sz, args.verbose);

  // set from args 
  params.verbose = args.verbose;
  params.coupleChannels = false;
  params.var_mode = (args.var_mode == 0) ? CLIPPED : PAUL_VAR;
  if (!args.use_default){

    int nsim = args.num_patches[index];

    // Override with command line parameters
    if (args.sizeSearchWindow[index] >= 0)
      VideoNLB::setSizeSearchWindow(params, args.sizeSearchWindow[index]);
    if (args.num_patches[index] >= 0)
      VideoNLB::setNSimilarPatches(params, args.num_patches[index]);

    // float values for alg 
    float thresh = args.thresh[index];
    params.variThres = (thresh > 0) ? thresh : params.variThres;
    params.rank = (args.rank[index] >= 0) ? args.rank[index] : params.rank;
    params.beta = (args.beta[index] > 0) ? args.beta[index] : params.beta;
    params.tau = (args.tau[index] > 0) ? args.tau[index] : params.tau;


    // optionally set these bools
    int aggreBoost = args.aggreBoost[index];
    params.aggreBoost = (aggreBoost != -1) ?  aggreBoost : params.aggreBoost;
    bool legal = (aggreBoost == -1) || (aggreBoost == 0) || (aggreBoost == 1);
    assert (legal == True);

    int procStep = args.procStep[index];
    params.procStep = (procStep != -1) ? procStep : params.procStep;
    legal = (procStep == -1) || (procStep >= 0);
    assert (legal == True);

    // always set 
    params.flatAreas = args.flat_areas[index];
    params.coupleChannels = args.couple_ch[index];
    params.sigmaBasic = args.sigmaBasic[index];

  }
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

