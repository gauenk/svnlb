#pragma once

#include <vnlb/cpp/pybind/interface.h>

/*******************************

      Testing and CPP File IO
      to verify exact
      numerical precision
      of Python API

*******************************/

struct ReadVideoParams {
ReadVideoParams() :
  // --> image details <--
  video_paths(""),
    first_frame(0),
    last_frame(0),
    frame_step(1),
    verbose(0) {}

  // -- image params --
  bool verbose;
  int first_frame,last_frame,frame_step;
  const char* video_paths;
};

void readVideoForVnlbCpp(const ReadVideoParams& args, const VnlbTensors& tensors);
void readVideoForFlowCpp(const ReadVideoParams& args, const VnlbTensors& tensors);
