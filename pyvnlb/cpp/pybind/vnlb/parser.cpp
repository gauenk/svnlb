#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <pyvnlb/cpp/flow/defaults.h>
#include <pyvnlb/cpp/vnlb/VideoNLBayes.hpp>

#include <pyvnlb/cpp/pybind/vnlb/interface.h>
#include <pyvnlb/cpp/pybind/vnlb/parser.h>


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

