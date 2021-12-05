

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

#include <vnlb/cpp/flow/defaults.h>
#include <vnlb/cpp/pybind/interface.h>
#include <vnlb/cpp/pybind/flow/interface.h>

extern "C" {
#include <vnlb/cpp/flow/tvl1flow_lib.h>
#include <vnlb/cpp/video_io/iio.h>
}


void runTV1Flow(const PyTvFlowParams& args, const VnlbTensors& tensors) {

  // unpack shape
  int w = tensors.w;
  int h = tensors.h;
  int c = tensors.c;
  int t = tensors.t;

  // shorten names
  float *burst = tensors.noisy;
  float *fflow = tensors.fflow;
  float *bflow = tensors.bflow;

  // set flow by direction
  float *flow;
  if (args.direction == 0){
    flow = fflow;
  }else if(args.direction == 1){
    flow = bflow;
  }else{
    VNLB_THROW_MSG("invalid flow direction.");
  }


  // set params
  tvFlowParams params;
  setTvFlowParams(args,params);

  // correct pyramid size
  const float N = 1 + log(hypot(w, h)/16.0) / log(1/params.zfactor);
  if (N < params.nscales)
    params.nscales = N;
  if (params.nscales < params.fscale)
    params.fscale = params.nscales;

  // verbose printing
  if (params.verbose){
    fprintf(stdout,"height=%d width=%d nframes=%d\n",h,w,t);
    fprintf(stderr,
	    "nproc=%d tau=%f lambda=%f theta=%f nscales=%d "
	    "zfactor=%f nwarps=%d epsilon=%g direction=%d\n",
	    params.nproc, params.tau, params.lambda,
	    params.theta, params.nscales, params.zfactor,
	    params.nwarps, params.epsilon, args.direction);
  }

  // run flows per image
  int hwc = h*w*c;
  for (int _t = 0; _t < (t-1); ++_t){

    // message
    if (params.verbose){
      fprintf(stdout,"Computing flow %d/%d\n",_t+1,t-1);
    }

    // pick offsets
    int mult1 = (args.direction == 0) ? _t : (_t+1);
    int mult2 = (args.direction == 0) ? (_t+1) : _t;

    // point to image pairs
    float* image1 = burst + mult1*hwc;
    float* image2 = burst + mult2*hwc;

    // point to flow
    float *u = flow + _t*(h*w*2);
    float *v = u + h*w;

    //compute the optical flow
    Dual_TVL1_optic_flow_multiscale(image1, image2, u, v,
				    w, h, params.tau,
				    params.lambda, params.theta,
				    params.nscales, params.fscale,
				    params.zfactor, params.nwarps,
				    params.epsilon, params.verbose);

  }

}
