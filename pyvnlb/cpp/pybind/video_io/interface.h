#pragma once 


/*******************************

      Testing and CPP File IO 
      to verify exact
      numerical precision
      of Python API

*******************************/

struct ReadVideoParams {
ReadVideoParams() :
  // --> image details <--
  t(0),h(0),w(0),c(0),
    read_video(nullptr),
    video_paths(""),
    first_frame(0),
    last_frame(0),
    frame_step(1),
    verbose(0) {}
  
  // -- image params --
  bool verbose;
  int t,h,w,c;
  int first_frame,last_frame,frame_step;
  float* read_video;
  const char* video_paths;
};

void readVideoForVnlbCpp(const ReadVideoParams& args);
void readVideoForFlowCpp(const ReadVideoParams& args);
