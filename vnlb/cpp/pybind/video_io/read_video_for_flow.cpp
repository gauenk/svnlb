
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

void readVideoForFlowCpp(const ReadVideoParams& args, const VnlbTensors& tensors) {

  // prints
  if (args.verbose){
    fprintf(stdout,"-- [readVideoForFlow] Parameters --\n");
    fprintf(stdout,"video_paths: %s\n",args.video_paths);
    fprintf(stdout,"first_frame: %d\n",args.first_frame);
    fprintf(stdout,"last_frame: %d\n",args.last_frame);
    fprintf(stdout,"frame_step: %d\n",args.frame_step);
    fprintf(stdout,"(t,h,w): (%d,%d,%d)\n",tensors.t,tensors.h,tensors.w);
  }

  // load cpp video
  int size = tensors.w*tensors.h;
  for(int tidx = args.first_frame;
      tidx <= args.last_frame;
      tidx += args.frame_step){
    int w, h;
    char filename[1024];
    sprintf(filename,args.video_paths,tidx);
    float* dataPtr = tensors.noisy+tidx*size; // just use "noisy" as container
    float* iioPtr = iio_read_image_float(filename, &w, &h);
    std::memcpy(dataPtr,iioPtr,w*h*sizeof(float));
    assert(w == tensors.w);
    assert(h == tensors.h);
  }

}
