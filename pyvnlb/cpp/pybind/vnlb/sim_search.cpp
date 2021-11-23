
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

void print_message(){
  // fprintf(stdout,"groupSize: %d\n",groupSize);
  // fprintf(stdout,"patch_num: %d\n",patch_num);
  // fprintf(stdout,"patch_dim: %d\n",patch_dim);
  // fprintf(stdout,"patch_chnls: %d\n",patch_chnls);
  // fprintf(stdout,"sWx,sWt: %d,%d\n",sWx,sWt);
  // fprintf(stdout,"step1: %d\n",step1);
}


void runSimSearch(VideoNLB::nlbParams& params,
                  const VnlbTensors& tensors,
                  PySimSearchParams& sim_params) {

  /*****************

   initialize tensors

  *****************/
  if (params.verbose){
    VideoNLB::printNlbParameters(params);
  }

  // unpack sim params
  unsigned nParts = sim_params.nParts;
  unsigned& nSimP = sim_params.nSimP;
  unsigned pidx = sim_params.pidx;
  bool all_pix = sim_params.all_pix;


  // unpack shape
  int w = tensors.w;
  int h = tensors.h;
  int c = tensors.c;
  int t = tensors.t;

  // setup video data
  Video<float> imNoisy,imNoisyRGB,imBasic,imClean,fflow,bflow;
  imNoisy.loadVideoFromPtr(tensors.noisy,w,h,c,t);
  imNoisyRGB.loadVideoFromPtr(tensors.noisy,w,h,c,t);
  if (params.isFirstStep){
    imBasic.loadVideoFromPtr(tensors.basic,w,h,c,t);
  } else{
    imBasic.loadVideoFromPtr(tensors.noisy,w,h,c,t);
  }
  if (tensors.use_clean){
    imClean.loadVideoFromPtr(tensors.clean,w,h,c,t);
  }
  if (tensors.use_flow){
    fflow.loadVideoFromPtr(tensors.fflow,w,h,2,t);
    bflow.loadVideoFromPtr(tensors.bflow,w,h,2,t);
  }

  // for(int k = 0; k < 10; ++k){
  //   fprintf(stdout,"imNoisy[%d]: %2.2f\n",k,imNoisy.data[k]);
  //   // fprintf(stdout,"fflow[%d]: %2.2f\n",k,fflow.data[k]);
  //   // fprintf(stdout,"bflow[%d]: %2.2f\n",k,bflow.data[k]);
  // }

  //
  // initialization for output vars
  //

  // unpack variables used for allocation
  const bool step1 = params.isFirstStep;
  const unsigned sWx = params.sizeSearchWindow;
  const unsigned sWt = params.sizeSearchTimeFwd +
    params.sizeSearchTimeBwd + 1;// VIDEO
  const unsigned sPx = params.sizePatch;
  const unsigned sPt = params.sizePatchTime;
  const VideoSize sz = imNoisy.sz;
  const unsigned patch_dim = sPx * sPx * sPt * (params.coupleChannels ? sz.channels : 1);
  const unsigned patch_chnls = params.coupleChannels ? 1 : sz.channels;
  const unsigned patch_num = sWx * sWx * sWt;

  // initialize vectors
  int groupSize = patch_num * patch_dim * patch_chnls;
  fprintf(stdout,"groupSize: %d\n",groupSize);
  fprintf(stdout,"patch_num: %d\n",patch_num);

  /*****************

  sub-divide (or "tile") the videos

  **********************/

  // Borders added to each sub-division of the image (for multi-threading)
  const int border = 2*(params.sizeSearchWindow/2)+ params.sizePatch - 1;
  fprintf(stdout,"border: %d\n",border);

  // color transform
  VideoUtils::transformColorSpace(imNoisy, true);
  VideoUtils::transformColorSpace(imClean, true);
  if (imBasic.sz.whcf > 0) VideoUtils::transformColorSpace(imBasic, true);

  // Split optical flow
  std::vector<Video<float> > fflowSub(nParts), bflowSub(nParts);
  std::vector<VideoUtils::TilePosition > oflowCrops(nParts);
  VideoUtils::subDivideTight(fflow, fflowSub, oflowCrops, border, nParts);
  VideoUtils::subDivideTight(bflow, bflowSub, oflowCrops, border, nParts);

  // Divide the noisy image into sub-images in order to easier parallelize
  std::vector<Video<float> > imNoisySub(nParts);
  std::vector<Video<float> > imNoisyRGBSub(nParts);
  std::vector<Video<float> > imCleanSub(nParts);
  std::vector<Video<float> > imBasicSub(nParts);
  std::vector<VideoUtils::TilePosition > imCrops(nParts);
  VideoUtils::subDivideTight(imNoisy, imNoisySub, imCrops, border, nParts);
  VideoUtils::subDivideTight(imNoisyRGB, imNoisyRGBSub, imCrops, border, nParts);
  VideoUtils::subDivideTight(imClean, imCleanSub, imCrops, border, nParts);
  VideoUtils::subDivideTight(imBasic, imBasicSub, imCrops, border, nParts);

  // run for each part
  for (int part = 0; part < nParts; ++part){

    // run for the part
    // float* start = tensors.groupNoisy;
    // int size = groupSize;
    std::vector<float> groupNoisy(groupSize);//start,start+size);

    // start = tensors.groupBasic;
    int size = step1 ? 0 : groupSize;
    std::vector<float> groupBasic(size);//start,start+size);

    // unsigned* istart = tensors.indices;
    // size = patch_num;
    std::vector<unsigned> indices(patch_num);//istart,istart+size);

    // exec similarity search
    if ((pidx >= 0) && (!all_pix)){
      nSimP = estimateSimilarPatches(imNoisySub[part],imBasicSub[part],
                                     fflowSub[part],bflowSub[part],
                                     groupNoisy,groupBasic,indices,
                                     pidx,params,imCleanSub[part],
                                     imNoisyRGBSub[part]);
    // }else if (all_pix){
    //   for (int idx=0; idx < sz.whcf; ++idx){
    //     nSimP += estimateSimilarPatches(imNoisy,imBasic,fflow,bflow,
    //                                     groupNoisy,groupBasic,indices,
    //                                     idx,params,imClean);
    //   }
    }else{
      fprintf(stdout,"check your inputs. "
              "[sim_search.cpp] does not know what to do\n");
    }

    // write back to tensors
    int gStep = part * groupSize;
    float* dataPtr = groupNoisy.data();
    std::fprintf(stdout,"gStep: %d | groupNoisy.size(): %ld\n",gStep,groupNoisy.size());
    std::memcpy(tensors.groupNoisy+gStep,dataPtr,groupNoisy.size());
    dataPtr = groupBasic.data();
    std::memcpy(tensors.groupBasic+gStep,dataPtr,groupBasic.size());
    // groupNoisy.saveVideoToPtr(tensors.groupNoisy);
    // groupBasic.saveVideoToPtr(tensors.groupBasic);

    int iStep = part * patch_num;
    unsigned* idataPtr = indices.data();
    std::memcpy(tensors.indices+iStep,idataPtr,indices.size());

  }


}

