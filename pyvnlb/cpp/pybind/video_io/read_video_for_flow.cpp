
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

#include <pyvnlb/cpp/utils/VnlbAsserts.h>
#include <pyvnlb/cpp/vnlb/VideoNLBayes.hpp>
#include <pyvnlb/cpp/pybind/video_io/interface.h>

extern "C" {
#include <pyvnlb/cpp/flow/tvl1flow_lib.h>
#include <pyvnlb/cpp/video_io/iio.h>
}



/*******************************

      Testing and CPP File IO 
      to verify exact
      numerical precision
      of Python API

*******************************/

void readVideoForFlowCpp(const ReadVideoParams& args) {

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
