
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

#include <vnlb/cpp/lib/VnlbAsserts.h>
#include <vnlb/cpp/src/pybind/py_res.h>
#include <vnlb/cpp/src/pybind/py_params.h>
#include <vnlb/cpp/src/VNLBayes/VideoNLBayes.hpp>
#include <vnlb/cpp/lib/tvl1flow/tvl1flow_lib.h>

extern "C" {
#include <vnlb/cpp/lib/iio/iio.h>
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
  // oracle.loadVideoFromPtr(args.oracle,w,h,c,t);
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
  auto tmp = params2.sizePatch;
  params2.sizePatch = 0;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  if (args.testing){
    basic.saveVideoToPtr(const_cast<float*>(args.basic));
  }
  if (args.verbose)
    printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
		       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);

  params1.sizePatch = 0;
  params2.sizePatch = tmp;
  // noisy.loadVideoFromPtr(args.noisy,w,h,c,t);
  groupsRatio = VideoNLB::runNLBayesThreads(noisy, fflow, bflow, basic, final,
  					    params1, params2, oracle);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  
  std::cout << "finished computation at " << std::ctime(&end_time)
	    << "elapsed time: " << elapsed_seconds.count() << "s\n";

  if (args.verbose)
    printf("Done. Processed %5.2f%% of possible patch groups in 1st step, and\n"
		       "%5.2f%% in 2nd step.\n", groupsRatio[0], groupsRatio[1]);
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

/*******************************

      Testing and CPP File IO 
      to verify exact
      numerical precision
      of Python API

*******************************/

void readVideoForVnlb(const ReadVideoParams& args) {

  // prints
  if (args.verbose){
    fprintf(stdout,"-- [readVideoForVnlb] Parameters --\n");
    fprintf(stdout,"video_paths: %s\n",args.video_paths);
    fprintf(stdout,"first_frame: %d\n",args.first_frame);
    fprintf(stdout,"last_frame: %d\n",args.last_frame);
    fprintf(stdout,"frame_step: %d\n",args.frame_step);
    fprintf(stdout,"(t,c,h,w): (%d,%d,%d,%d)\n",args.t,args.c,args.h,args.w);
  }
  
  // init videos 
  Video<float> cppVideo,pyVideo;
  cppVideo.loadVideo(args.video_paths,args.first_frame,args.last_frame,args.frame_step);
  float* cppPtr = cppVideo.data.data();
  int size = cppVideo.sz.whcf;
  std::memcpy(args.read_video,cppPtr,size*sizeof(float));

}

void readVideoForFlow(const ReadVideoParams& args) {

  // prints
  if (args.verbose){
    fprintf(stdout,"-- [readVideoForFlow] Parameters --\n");
    fprintf(stdout,"video_paths: %s\n",args.video_paths);
    fprintf(stdout,"first_frame: %d\n",args.first_frame);
    fprintf(stdout,"last_frame: %d\n",args.last_frame);
    fprintf(stdout,"frame_step: %d\n",args.frame_step);
    fprintf(stdout,"(t,h,w): (%d,%d,%d)\n",args.t,args.h,args.w);
  }

  // load cpp video
  int size = args.w*args.h;
  for(int tidx = args.first_frame;
      tidx <= args.last_frame;
      tidx += args.frame_step){
    int w, h;
    char filename[1024];
    sprintf(filename,args.video_paths,tidx);
    float* dataPtr = args.read_video+tidx*size;
    float* iioPtr = iio_read_image_float(filename, &w, &h);
    std::memcpy(dataPtr,iioPtr,w*h*sizeof(float));
    assert(w == args.w);
    assert(h == args.h);
  }

}


// void testIIORead(const ReadVideoParams& args) {
  
//   // init videos 
//   Video<float> cppVideo,pyVideo;
//   if (args.verbose){
//     fprintf(stdout,"-- testIIORead Parameters --\n");
//     fprintf(stdout,"video_paths: %s\n",args.video_paths);
//     fprintf(stdout,"first_frame: %d\n",args.first_frame);
//     fprintf(stdout,"last_frame: %d\n",args.last_frame);
//     fprintf(stdout,"frame_step: %d\n",args.frame_step);
//     fprintf(stdout,"(t,h,w): (%d,%d,%d)\n",args.t,args.h,args.w);
//   }

//   // load python video 
//   assert(args.c == 1);
//   pyVideo.loadVideoFromPtr(args.test_video,args.w,args.h,args.c,args.t);
//   cppVideo.resize(pyVideo.sz);

//   // load cpp video
//   int size = pyVideo.sz.whc;
//   float* cppVecPtr = cppVideo.data.data();
//   for(int tidx = args.first_frame;
//       tidx <= args.last_frame;
//       tidx += args.frame_step){
//     int w, h;
//     char filename[1024];
//     sprintf(filename,args.video_paths,tidx);
//     float* cppPtr = cppVecPtr+tidx*size;
//     float* iioPtr = iio_read_image_float(filename, &w, &h);
//     std::memcpy(cppPtr,iioPtr,w*h*sizeof(float));
//     assert(w == args.w);
//     assert(h == args.h);
//   }

//   // // compute difference 
//   // float delta = 0;
//   // auto cppData = cppVideo.data;
//   // auto pyData = pyVideo.data;
//   // for (int i = 0; i < size; ++i){
//   //   float delta_i = (cppData[i]-pyData[i]);
//   //   delta += (delta_i * delta_i);
//   // }

//   // // print report 
//   // if (args.verbose){
//   //   fprintf(stdout,"[Cpp v. Python]: %2.3e\n",delta);
//   // }
//   // *args.delta = delta;

// }
