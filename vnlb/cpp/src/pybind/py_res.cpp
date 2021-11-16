
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>
#include <float.h>

#include <vnlb/cpp/lib/VnlbAsserts.h>
#include <vnlb/cpp/src/pybind/py_res.h>
#include <vnlb/cpp/src/pybind/py_params.h>
#include <vnlb/cpp/src/VNLBayes/VideoNLBayes.hpp>
#include <vnlb/cpp/lib/iio/iio.h>
#include <vnlb/cpp/lib/tvl1flow/tvl1flow_lib.h>

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
    bflow.loadVideoFromPtr(args.bflow,w,h,2,t);
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
  if (args.testing){
    basic.saveVideoToPtr(const_cast<float*>(args.basic));
  }
  // final.saveVideo("deno_%03d.png", 0, 1);
  // noisy.saveVideo("noisy_%03d.png", 0, 1);

}


void runTV1Flow(const PyTvFlowParams& args) {
  
  
  // unpack shape 
  int w = args.w;
  int h = args.h;
  int c = args.c;
  int t = args.t;

  // remove const cast needed (i think) by SWIG-Python
  float *burst = const_cast<float*>(args.burst);
  float *fflow = const_cast<float*>(args.fflow);
  float *bflow = const_cast<float*>(args.bflow);

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

  // flow for the burst
  // char name[] = "flow_";
  // char ending[] = ".flo";
  // char outfile[1024];

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

    //save the optical flow
    // ostringstream temp;
    // temp << (_t+1);
    // auto str = temp.str();
    // auto result = name + str + ending;
    // strcpy(outfile,result.c_str());
    
    // iio_save_image_float_split(outfile, u, w, h, 2);

  }

}

