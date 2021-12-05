

#pragma once

#ifndef DUAL_TVL1_OPTIC_FLOW_H
#define DUAL_TVL1_OPTIC_FLOW_H

#include <float.h>


void Dual_TVL1_optic_flow_multiscale(
float *I0,           // source image
float *I1,           // target image
float *u1,           // x component of the optical flow
float *u2,           // y component of the optical flow
const int   nxx,     // image width
const int   nyy,     // image height
const float tau,     // time step
const float lambda,  // weight parameter for the data term
const float theta,   // weight parameter for (u - v)²
const int   nscales, // number of scales
const int   fscale , // finer scale (drop the scales finer than this one)
const float zfactor, // factor for building the image piramid
const int   warps,   // number of warpings per scale
const float epsilon, // tolerance for numerical convergence
const bool  verbose  // enable/disable the verbose mode
);

void Dual_TVL1_optic_flow(
float *I0,           // source image
float *I1,           // target image
float *u1,           // x component of the optical flow
float *u2,           // y component of the optical flow
const int   nx,      // image width
const int   ny,      // image height
const float tau,     // time step
const float lambda,  // weight parameter for the data term
const float theta,   // weight parameter for (u - v)²
const int   warps,   // number of warpings per scale
const float epsilon, // tolerance for numerical convergence
const bool  verbose  // enable/disable the verbose mode
);

void image_normalization(
const float *I0,  // input image0
const float *I1,  // input image1
float *I0n,       // normalized output image0
float *I1n,       // normalized output image1
int size          // size of the image
);

float energy_optic_flow(
float *I0,           // source image
float *I1,           // target image
float *u1,           // x component of the optical flow
float *u2,           // y component of the optical flow
float *diff,         // difference between I0 and I1 after warp
const int   nx,      // image width
const int   ny,      // image height
const float lambda   // weight parameter for the data term
);


static void getminmax(
	float *min,     // output min
	float *max,     // output max
	const float *x, // input array
	int n           // array size
);


#endif//DUAL_TVL1_OPTIC_FLOW_H
