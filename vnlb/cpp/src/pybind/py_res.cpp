
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <vnlb/cpp/src/pybind/py_res.h>
#include <vnlb/cpp/src/pybind/py_params.h>
#include <vnlb/cpp/src/VNLBayes/VideoNLBayes.hpp>
#include <vnlb/cpp/lib/iio/iio.h>
#include <vnlb/cpp/lib/tvl1flow/tvl1flow_lib.h>

using namespace std;

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
  // final.saveVideo("deno_%03d.png", 0, 1);
  // noisy.saveVideo("noisy_%03d.png", 0, 1);

}


void runTV1Flow(const PyTvFlowParams& args) {
  
  
  // unpack shape 
  int w = args.w;
  int h = args.h;
  int c = args.c;
  int t = 1;
  // std::fprintf(stdout,"(w,h,c,t): (%d,%d,%d,%d)\n",w,h,c,t);

  // remove const cast (i think) needed by SWIG-Python
  float *image1, *image2;
  image1 = const_cast<float*>(args.image1);
  image2 = const_cast<float*>(args.image2);

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
  if (params.verbose)
    fprintf(stderr,
	    "nproc=%d tau=%f lambda=%f theta=%f nscales=%d "
	    "zfactor=%f nwarps=%d epsilon=%g\n",
	    params.nproc, params.tau, params.lambda,
	    params.theta, params.nscales, params.zfactor,
	    params.nwarps, params.epsilon);

  //allocate memory for the flow
  float *u = const_cast<float*>(args.flow);
  float *v = u + h*w;

  //compute the optical flow
  Dual_TVL1_optic_flow_multiscale(image1, image2, u, v,
  				  w, h, params.tau,
  				  params.lambda, params.theta,
  				  params.nscales, params.fscale,
  				  params.zfactor, params.nwarps,
  				  params.epsilon, params.verbose);

  // [testing] save the optical flow
  // if (args.testing){
  //   char* outfile = "pyflow_%03d.flo";
  //   iio_save_image_float_split(outfile, u, w, h, 2);
  // }

}

