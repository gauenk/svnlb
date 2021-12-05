
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

#include <vnlb/cpp/pybind/interface.h>
#include <vnlb/cpp/pybind/video_io/interface.h>

extern "C" {
#include <vnlb/cpp/flow/tvl1flow_lib.h>
#include <vnlb/cpp/video_io/iio.h>
}


/*******************************

      Testing and CPP File IO
      to verify exact
      numerical precision
      of Python API

*******************************/

void readVideoForVnlbCpp(const ReadVideoParams& args, const VnlbTensors& tensors) {

  // prints
  if (args.verbose){
    fprintf(stdout,"-- [readVideoForVnlb] Parameters --\n");
    fprintf(stdout,"video_paths: %s\n",args.video_paths);
    fprintf(stdout,"first_frame: %d\n",args.first_frame);
    fprintf(stdout,"last_frame: %d\n",args.last_frame);
    fprintf(stdout,"frame_step: %d\n",args.frame_step);
    fprintf(stdout,"(t,c,h,w): (%d,%d,%d,%d)\n",tensors.t,tensors.c,tensors.h,tensors.w);
  }

  // init videos
  Video<float> cppVideo,pyVideo;
  cppVideo.loadVideo(args.video_paths,args.first_frame,args.last_frame,args.frame_step);
  float* cppPtr = cppVideo.data.data();
  int size = cppVideo.sz.whcf;
  std::memcpy(tensors.noisy,cppPtr,size*sizeof(float));

}
