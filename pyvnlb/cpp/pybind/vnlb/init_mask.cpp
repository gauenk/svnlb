
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

#include <pyvnlb/cpp/pybind/interface.h>
#include <pyvnlb/cpp/pybind/vnlb/interface.h>

extern "C" {
#include <pyvnlb/cpp/flow/tvl1flow_lib.h>
#include <pyvnlb/cpp/video_io/iio.h>
}

void init_mask_cpp(MaskParams params, int& ngroups){

  // --> some unpacking <--
  int nframes = params.nframes;
  int height = params.height;
  int width = params.width;
  int sPx = params.ps;
  int sPt = params.ps_t;
  int sWx = params.sWx;
  int sWt = params.sWt;
  int step_t = params.step_t;
  int step_h = params.step_h;
  int step_w = params.step_w;


  /***********************

     Derived Constants

  ************************/

  // There's a border added only if the crop doesn't touch the source image border
  bool border_w0 = params.origin_w > 0;
  bool border_h0 = params.origin_h > 0;
  bool border_t0 = params.origin_t > 0;
  bool border_w1 = params.ending_w < width;
  bool border_h1 = params.ending_h < height;
  bool border_t1 = params.ending_t < nframes;

  // Origin and end of processing region (border excluded)
  int border_s = sPx-1 + sWx/2;
  int border_t = sPt-1 + sWt/2;
  int ori_w =                        border_w0 ? border_s : 0 ;
  int ori_h =                        border_h0 ? border_s : 0 ;
  int ori_t =                        border_t0 ? border_t : 0 ;
  int end_w = (int)width  - (int)(border_w1 ? border_s : sPx-1);
  int end_h = (int)height - (int)(border_h1 ? border_s : sPx-1);
  int end_t = (int)nframes - (int)(border_t1 ? border_t : sPt-1);

  /***********************

        Fill Mask

  ************************/

  // --> init mask <--
  ngroups = 0;
  Video<char> mask(width,height,nframes,1,false);

  // --> fill mask <--
  for (int t = ori_t, dt = 0; t < end_t; t++, dt++){
    for (int h = ori_h, dh = 0; h < end_h; h++, dh++){
      for (int w = ori_w, dw = 0; w < end_w; w++, dw++){
          if ( (dt % step_t == 0) || (!border_t1 && t == end_t - 1)){

              int phase_h = (!border_t1 && t == end_t - 1) ? 0 : t/step_t;

              if ( (dh % step_h == phase_h % step_h) ||
                   (!border_h1 && h == end_h - 1) ||
                   (!border_h0 && h == ori_h    ) )
                {
                  int phase_w = (!border_h1 && h == end_h - 1) ? 0 : (phase_h+h/step_h);

                  if ( (dw % step_w == phase_w % step_w) ||
                       (!border_w1 && w == end_w - 1) ||
                       (!border_w0 && w == ori_w    ) )
                    {
                      mask(w,h,t) = true;
                      ngroups++;
                    }
                }
          }
      }
    }
  }

  // --> write back to python <--
  std::copy(mask.data.begin(),mask.data.end(),(char*)params.mask);

}
