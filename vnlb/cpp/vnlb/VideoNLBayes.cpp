/* Modified work: Copyright (c) 2019, Pablo Arias <pariasm@gmail.com>
 * Original work: Copyright (c) 2013, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * 
 * This program is free software: you can use, modify and/or redistribute it
 * under the terms of the GNU Affero General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version. You should have received a copy of this license
 * along this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */

/* VideoNLBayes.cpp
 * Implementation of the Video NL-Bayes denoising algorithm
 */

#include <stdexcept>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <float.h>


#include <stdio.h>     // getchar() for debugging

#include "vnlb_params.h"
#include "VideoNLBayes.hpp"
#include "LibMatrix.h"


#ifdef _OPENMP
#include <omp.h>
#endif

// colors
#define ANSI_BLK  "\x1b[30m"
#define ANSI_RED  "\x1b[31m"
#define ANSI_GRN  "\x1b[32m"
#define ANSI_YLW  "\x1b[33m"
#define ANSI_BLU  "\x1b[34m"
#define ANSI_MAG  "\x1b[35m"
#define ANSI_CYN  "\x1b[36m"
#define ANSI_WHT  "\x1b[37m"
#define ANSI_BBLK "\x1b[30;01m"
#define ANSI_BRED "\x1b[31;01m"
#define ANSI_BGRN "\x1b[32;01m"
#define ANSI_BYLW "\x1b[33;01m"
#define ANSI_BBLU "\x1b[34;01m"
#define ANSI_BMAG "\x1b[35;01m"
#define ANSI_BCYN "\x1b[36;01m"
#define ANSI_BWHT "\x1b[37;01m"
#define ANSI_RST  "\x1b[0m"


namespace VideoNLB
{

void defaultParameters(
	nlbParams &prms,
	int psz_x,
	int psz_t,
	const unsigned step,
	const float sigma,
	const VideoSize &size,
	const bool verbose)
{
	const bool s1 = (step == 1);
	const float s = sigma;
	const unsigned channels = size.channels;

	using std::max;
	using std::min;

    // default is "use max num of threads"
    prms.nThreads = prms.set_nThreads ? prms.nThreads : 0;
    prms.nParts = prms.set_nParts ? prms.nParts : 0;

	// Default patch size
	if (psz_x == -1 || psz_t == -1)
	{
		psz_x = (channels   == 1) ? 10 : 7 ;
		psz_t = (size.frames > 1) ? 2  : 1 ;
	}

	// Store noise level in parameter struct
	prms.sigma = s;

	// Which step are we running?
	prms.isFirstStep = s1;

	// Patch size and search region
	prms.sizePatch         = psz_x; // Spatial patch size
	prms.sizePatchTime     = psz_t; // Temporal patch size
	prms.sizeSearchWindow  = 27;    // Spatial search diameter
	prms.sizeSearchTimeFwd = 6;     // Number of search frames (forward)
	prms.sizeSearchTimeBwd = 6;     // Number of search fraems (backward)

	// Maximum patch distance threshold for similar patches
	if(s1) prms.tau = 0; // 0 means that the threshold is not used
	else   prms.tau = 400;

	// Bayesian filtering parameters
	prms.rank = 39;    // Maximum rank of covariance matrices
	prms.beta = 1.f;   // Noise multiplier parameter when estimating

	// Distinct handling of flat areas
	prms.flatAreas = s1 ? false : true; // Toggle flat area trick
#ifdef FAT_ORIGINAL
	prms.gamma = 1.05f; // Parameter used to determine if an area is flat
#else
	prms.gamma = 0.95f; // Parameter used to determine if an area is flat
#endif

	// Parameters for speed-up by skipping patches
	prms.procStep     = max(1, psz_x / 2); // Step for skipping patches
	prms.aggreBoost = true; // Skip patches in a 4-neighborhood of already denoised patches

	// Print information?
	prms.verbose = verbose;

	// This is used to process a single frame (only useful for 2nd step)
	prms.onlyFrame = -1;

	// The number of similar patches and the variance threshold depend on the noise,
	// the number of channels, and on the chosen patch size
	if (size.channels == 1)
	{
		// Grayscale default parameters
		if (psz_x == 7 && psz_t == 4)
		{
			prms.nSimilarPatches = s1 ? 150 : 60;
			prms.variThres = s1 ? 1.1 + (s - 6.)*(2.9 - 2.1)/(48. - 6.)
			                    : 1.7 + (s - 6.)*(0.7 - 1.7)/(48. - 6.);
		}
		else if (psz_x == 8 && psz_t == 3)
		{
			prms.nSimilarPatches = s1 ? 150 : 60;
			prms.variThres = s1 ? 2.7 : 1.7 + (s - 6.)*(1.2 - 1.7)/(24. - 6.);
		}
		else if (psz_x == 10 && psz_t == 2)
		{
			prms.nSimilarPatches = s1 ? 150 : 60;
			prms.variThres = s1 ? 2.7 : 1.7 + (s - 6.)*(1.2 - 1.7)/(24. - 6.);
		}
		else if (psz_x == 14 && psz_t == 1)
		{
			prms.nSimilarPatches = s1 ? 150 : 40;
			prms.variThres = s1 ? 1.7 : 2.7 + (s - 6.)*(2.2 - 2.7)/(24. - 6.);
		}
		else if (psz_x == 8 && psz_t == 2)
		{
			prms.nSimilarPatches = s1 ? round(max(100.,200. + (s - 6.)*(150. - 200.)/(24. - 6.))) : 40;
			prms.variThres = s1 ? min(2.7, 1.1 + (s - 6.)*(2.9 - 2.1)/(24. - 6.))
			                    : max(0.7, 2.2 + (s - 6.)*(1.2 - 2.2)/(24. - 6.));
		}
		else if (psz_t >= 4)
		{
			// parameters tuned for a patch size of 5x5x4
			prms.nSimilarPatches = s1 ? 200 : round(40. + (s - 6.)*(60. - 40.)/(48. - 6.));
			prms.variThres = s1 ? 1.1 : max(0.5, 1.7 + (s - 6.)*(1.2 - 1.7)/(24. - 6.));

			if (prms.verbose && psz_x != 5)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 5x5x4.\n");
		}
		else if (psz_t == 3)
		{
			// parameters tuned for a patch size of 6x6x3
			prms.nSimilarPatches = s1 ? 200 : round(35. + (s - 6.)*(60. - 35.)/(48. - 6.));
			prms.variThres = s1 ? 1.1 : max(0.5,  2.2 + (s - 6.)*(1.2 - 2.2)/(24. - 6.));

			if (prms.verbose && psz_x != 6)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 6x6x3.\n");
		}
		else if (psz_t == 2)
		{
			// parameters tuned for a patch size of 7x7x2
			prms.nSimilarPatches = s1 ? 150 : round( 40. + (s - 6.)*(60. - 40.)/(48. - 6.) );
			prms.variThres = s1 ? 1.1 : max(0.5, 2.2 + (s - 6.)*(1.2 - 2.2)/(24. - 6.));

			if (prms.verbose && psz_x != 7)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 7x7x2.\n");
		}
		else if (psz_t == 1)
		{
			// parameters tuned for a patch size of 10x10x2
			prms.nSimilarPatches = s1 ? 100 : round( 40. + (s - 6.)*(60. - 40.)/(48. - 6.) );
			prms.variThres = s1 ? 1.9 : max(0.7, 2.2 + (s - 6.)*(1.2 - 2.2)/(24. - 6.));

			if (prms.verbose && psz_x && psz_x != 10)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 10x10x1.\n");
		}
	}
	else
	{
		// Default parameters for rgb videos
		if (psz_t >= 3)
		{
			// parameters tuned for a patch size of 5x5x4
			prms.nSimilarPatches = s1 ? 150 : 60;
			prms.variThres = s1 ? 1.9 : max(0.2, 1.7 + (1.2 - 1.7)*(s - 10)/(20 - 10));

			if (prms.verbose && psz_x != 5)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 5x5x4.\n");
		}
		else if (psz_t == 2)
		{
			// parameters tuned for a patch size of 7x7x2
			prms.nSimilarPatches = s1 ? 100 : 60;
			prms.variThres = s1 ? 2.7 : max(0.2, 1.7 + (1.2 - 1.7)*(s - 10)/(20 - 10));

			if (prms.verbose && psz_x != 7)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 7x7x2.\n");
		}
		else if (psz_t == 1)
		{
			// parameters tuned for a patch size of 10x10x1
			prms.nSimilarPatches = s1 ? 200 : 80;
			prms.variThres = s1 ? 1.9 : max(0., 0.7 + (0.2 - 0.7)*(s - 10)/(20 - 10));

			if (prms.verbose && psz_x != 10)
				fprintf(stderr, "Warning: No default params for given patch size. Keeping given\n"
				                "         patch size, but using rest of parameters for 10x10x1.\n");
		}
	}
}

void setSizeSearchWindow(nlbParams& prms, unsigned sizeSearchWindow)
{
	prms.sizeSearchWindow = sizeSearchWindow;
	prms.nSimilarPatches = std::min(prms.nSimilarPatches, sizeSearchWindow *
	                                                      sizeSearchWindow *
	                                                     (prms.sizeSearchTimeFwd +
	                                                      prms.sizeSearchTimeBwd + 1));
}

void setNSimilarPatches(nlbParams& prms, unsigned nSimilarPatches)
{
	prms.nSimilarPatches = nSimilarPatches;
	prms.nSimilarPatches = std::min(nSimilarPatches, prms.sizeSearchWindow *
	                                                 prms.sizeSearchWindow *
	                                                (prms.sizeSearchTimeFwd +
	                                                 prms.sizeSearchTimeBwd + 1));
}

void printNlbParameters(const nlbParams &prms)
{
	printf("\x1b[37;01m" "Parameters for step %d:" ANSI_RST "\n" , prms.isFirstStep ? 1 : 2);
	printf("\tPatch search:\n");
	printf("\t\tPatch size                  = %d\n"       , prms.sizePatch);
	printf("\t\tPatch size temporal         = %d\n"       , prms.sizePatchTime);
	printf("\t\tNumber of patches           = %d\n"       , prms.nSimilarPatches);
	if (!prms.isFirstStep) printf("\t\tDistance threshold (tau)    = %g\n"       , prms.tau);
	else                     printf("\t\tDistance threshold (tau)    = N/A\n"      );
	printf("\t\tSpatial search window       = %dx%d\n"    , prms.sizeSearchWindow, prms.sizeSearchWindow);
	printf("\t\tTemporal search range       = [-%d,%d]\n" , prms.sizeSearchTimeBwd, prms.sizeSearchTimeBwd);
	printf("\tGroup filtering:\n");
	printf("\t\tBeta                        = %g\n"       , prms.beta);
	printf("\t\tRank                        = %d\n"       , prms.rank);
	printf("\t\tChannels coupled            = %s\n"       , prms.coupleChannels ? "true" : "false");
	printf("\t\tVariance th.                = %g\n"       , prms.variThres);
	if (!prms.isFirstStep)
		printf("\t\tSigma basic                 = %g\n"       , prms.sigmaBasic);
	if (prms.flatAreas)
		printf("\t\tFlat area trick with gamma  = %g\n"       , prms.gamma);
	else
		printf("\t\tFlat area trick             = inactive\n");
	printf("\tSpeed-ups:\n");
	printf("\t\tProcessing step             = %d\n"       , prms.procStep);
	printf("\t\tAggregation boost           = %s\n\n"     , prms.aggreBoost ? "active" : "inactive");
	if (prms.onlyFrame >= 0)
		printf("\tProcessing only frame %d\n\n", prms.onlyFrame);
}

std::vector<float> runNLBayesThreads(
	Video<float> & imNoisy,
	Video<float> const& fflow,
	Video<float> const& bflow,
	Video<float> &imBasic,
	Video<float> &imFinal,
	const nlbParams prms1,
	const nlbParams prms2,
	Video<float> &imClean)
{
	// Only 1, 3 or 4-channels images can be processed.
	const unsigned chnls = imNoisy.sz.channels;
	if (! (chnls == 1 || chnls == 3 || chnls == 4))
		throw std::runtime_error("VideoNLB::runNLBayes: Wrong number of "
				"channels. Must be 1, 3 or 4.");

	// Check if optical flow is valid
	const bool use_oflow = (fflow.sz.width > 0);
	if (use_oflow &&
	   (fflow.sz.channels != 2 ||
	    fflow.sz.width  != imNoisy.sz.width  ||
	    fflow.sz.height != imNoisy.sz.height ||
	    fflow.sz.frames != imNoisy.sz.frames))
		throw std::runtime_error("VideoNLB::runNLBayes: Wrong size of fwd optical flow.");

	if (use_oflow &&
	   (bflow.sz.channels != 2 ||
	    bflow.sz.width  != imNoisy.sz.width  ||
	    bflow.sz.height != imNoisy.sz.height ||
	    bflow.sz.frames != imNoisy.sz.frames))
		throw std::runtime_error("VideoNLB::runNLBayes: Wrong size of bwd optical flow.");

	// Print compiler options
	if (prms1.verbose)
	{
	  if(prms1.var_mode == FAT_OG){
	    printf(ANSI_BCYN
		   "NEGATIVE_VARIANCES > Allowing negative variances\n"
		   ANSI_RST);
	  }
	  else if(prms1.var_mode == PAUL_VAR){
	    printf(ANSI_BCYN
		   "PAUL_VARIANCE > Full inverse of Paul expected eigenvalue\n"
		   ANSI_RST);
	  }else if(prms1.var_mode == PAUL_SIMPLE){
	    printf(ANSI_BCYN "PAUL_SIMPLE_VARIANCE > Approximate inverse "
		   "of Paul expected eigenvalue\n" ANSI_RST);
	  }else if(prms1.var_mode == CLIPPED){
	    printf(ANSI_BCYN "CLIPPED_VARIANCE > "
		   "Only big eigs allowed.\n" ANSI_RST);
	  }
	}

	// Determine steps (0 means both steps, 1 iteration one only, 2 iter two only, -1 none)
	int steps = prms1.sizePatch ?
	           (prms2.sizePatch ? 0 : 1) :
	           (prms2.sizePatch ? 2 :-1) ;

	if (steps == -1)
		throw std::runtime_error("VideoNLB::runNLBayes: Both patch sizes are zero.");

	// Video size
	VideoSize size = imNoisy.sz;

	// Initialization
	if (steps != 2) imBasic.resize(size);
	if (steps != 1) imFinal.resize(size);

	if (steps == 2 && size != imBasic.sz)
		throw std::runtime_error("VideoNLB::runNLBayes: sizes of noisy and "
				"basic videos don't match");

	// Print parameters
	if (prms1.verbose) if (steps != 2) printNlbParameters(prms1);
	if (prms2.verbose) if (steps != 1) printNlbParameters(prms2);

	// Percentage of processed groups over total number of pixels
	std::vector<float> groupsRatio(2,0.f);

	// RGB to YUV
	VideoUtils::transformColorSpace(imNoisy, true);
	VideoUtils::transformColorSpace(imClean, true);
	if (steps == 2) VideoUtils::transformColorSpace(imBasic, true);

	// Multithreading: split video in tiles
    int maxThreads = -1;
	unsigned nThreads = 1;
#ifdef _OPENMP
	maxThreads = omp_get_max_threads();
    assert (prms1.nThreads == prms2.nThreads);
    if ( (prms1.nThreads > maxThreads) || (prms1.nThreads == 0) ){
      nThreads = maxThreads;
    }else{
      nThreads = prms1.nThreads;
      omp_set_num_threads(nThreads);
      // omp_set_dynamic(1);
    }
	if (prms1.verbose){
      printf(ANSI_CYN "OpenMP is using %d threads\n" ANSI_RST, nThreads);
    }
#endif

    // Parse num parts from paramters
	unsigned nParts = 2 * nThreads; // number of video parts
    assert(prms1.nParts == prms2.nParts);
    if (prms1.nParts > 0){
      nParts = prms1.nParts;
    }
    assert(nThreads <= nParts);
    // fprintf(stdout,"nThreads,nParts: %d,%d\n",nThreads,nParts);

	// Borders added to each sub-division of the image (for multi-threading)
	const int border = std::max(2*(prms1.sizeSearchWindow/2) + prms1.sizePatch - 1,
	                            2*(prms2.sizeSearchWindow/2) + prms2.sizePatch - 1);

	// Split optical flow
	std::vector<Video<float> > fflowSub(nParts), bflowSub(nParts);
	std::vector<VideoUtils::TilePosition > oflowCrops(nParts);
	VideoUtils::subDivideTight(fflow, fflowSub, oflowCrops, border, nParts);
	VideoUtils::subDivideTight(bflow, bflowSub, oflowCrops, border, nParts);

	// Divide the noisy image into sub-images in order to easier parallelize the process
	std::vector<Video<float> > imNoisySub(nParts);
	std::vector<Video<float> > imCleanSub(nParts);
	std::vector<Video<float> > imBasicSub(nParts);
	std::vector<VideoUtils::TilePosition > imCrops(nParts);
	VideoUtils::subDivideTight(imNoisy, imNoisySub, imCrops, border, nParts);
	VideoUtils::subDivideTight(imClean, imCleanSub, imCrops, border, nParts);
	VideoUtils::subDivideTight(imBasic, imBasicSub, imCrops, border, nParts);

	std::vector<Video<float> > imFinalSub(nParts);

	std::vector<nlbParams> prms(2);
	prms[0] = prms1;
	prms[1] = prms2;

	// Run VNLBayes steps
	for (int iter = 0; iter < 2; ++iter){
		if (prms[iter].sizePatch)
		{
			if (prms[iter].verbose)
			{
				if (iter == 0)
				{
					printf("1st Step\n");
					for (int p = 0; p < nParts; ++p) printf("\n");
				}
				else
				{
					if (steps == 2) for (int p = 0; p <= nParts; ++p) printf("\n");
					printf("\x1b[%dF2nd Step\n",nParts+1);
					for (int p = 0; p < nParts; ++p) printf("\x1b[2K\n");
				}
			}

			// Process all sub-images
			std::vector<unsigned> groupsProcessedSub(nParts);
#ifdef _OPENMP
			// we make a copy of prms structure because, since it is constant,
			// it causes a compilation error with OpenMP (only on IPOL server)
			nlbParams prms_cpy(prms[iter]);
#pragma omp parallel for schedule(dynamic, nParts/nThreads) \
			num_threads(4) \
			shared(imNoisySub, imBasicSub, imFinalSub) \
			firstprivate(prms_cpy)
#endif
			for (int n = 0; n < (int)nParts; n++){
              // fprintf(stdout,"iter,n: %d,%d\n",iter,n);
              // fprintf(stdout,"n: %d,%d,%d\n",n,
              //         imNoisySub[n].sz.width,imNoisySub[n].sz.whcf);
				groupsProcessedSub[n] =
					processNLBayes(imNoisySub[n], fflowSub[n], bflowSub[n],
					               imBasicSub[n], imFinalSub[n], prms[iter], imCrops[n],
					               imCleanSub[n]);
            }

			for (int n = 0; n < (int)nParts; n++)
				groupsRatio[iter] += 100.f * (float)groupsProcessedSub[n]/(float)size.whf;
		}
    }

	// Get the basic estimate
	VideoUtils::subBuildTight(imBasicSub, imBasic, border);
	VideoUtils::subBuildTight(imFinalSub, imFinal, border);

	// YUV to RGB
	VideoUtils::transformColorSpace(imNoisy, false);
	VideoUtils::transformColorSpace(imBasic, false);
	VideoUtils::transformColorSpace(imFinal, false);
	VideoUtils::transformColorSpace(imClean, false);

	return groupsRatio;
}

unsigned processNLBayes(
	Video<float> const& imNoisy,
	Video<float> const& fflow,
	Video<float> const& bflow,
	Video<float> &imBasic,
	Video<float> &imFinal,
	nlbParams const& params,
	VideoUtils::TilePosition crop,
	Video<float> const &imClean)
{
	using std::vector;

	// Parameters initialization
	const bool step1 = params.isFirstStep;
	const unsigned sWx = params.sizeSearchWindow;
	const unsigned sWt = params.sizeSearchTimeFwd +
	                     params.sizeSearchTimeBwd + 1;// VIDEO
	const unsigned sPx = params.sizePatch;
	const unsigned sPt = params.sizePatchTime;
	const VideoSize sz = imNoisy.sz;

	// Weight sum per pixel
	Video<float> weight(sz.width, sz.height, sz.frames, 1, 0.f);

	// Processing mask: true for pixels that need to be processed
	Video<char> mask(sz.width, sz.height, sz.frames, 1, false);

	// There's a border added only if the crop doesn't touch the source image border
    // fprintf(stdout,"crop.origin(x,y,t): (%d,%d,%d)\n",
    //         crop.origin_x,crop.origin_y,crop.origin_t);
    // fprintf(stdout,"crop.ending(x,y,t): (%d,%d,%d)\n",
    //         crop.ending_x,crop.ending_y,crop.ending_t);
    // fprintf(stdout,"crop.source_sz(w,h,f): (%d,%d,%d)\n",
    //         crop.source_sz.width,crop.source_sz.height,crop.source_sz.frames);
    // fprintf(stdout,"procStep: %d\n",params.procStep);

	bool border_x0 = crop.origin_x > 0;
	bool border_y0 = crop.origin_y > 0;
	bool border_t0 = crop.origin_t > 0;
	bool border_x1 = crop.ending_x < crop.source_sz.width;
	bool border_y1 = crop.ending_y < crop.source_sz.height;
	bool border_t1 = crop.ending_t < crop.source_sz.frames;

	int n_groups = 0;
	unsigned stepx = params.procStep;
	unsigned stepy = params.procStep;
	unsigned stepf = 1;

	// Origin and end of processing region (border excluded)
	int border_x = sPx-1 + sWx/2;
	int border_t = sPt-1 + sWt/2;
	int ori_x =                        border_x0 ? border_x : 0 ;
	int ori_y =                        border_y0 ? border_x : 0 ;
	int ori_f =                        border_t0 ? border_t : 0 ;
	int end_x = (int)sz.width  - (int)(border_x1 ? border_x : sPx-1);
	int end_y = (int)sz.height - (int)(border_y1 ? border_x : sPx-1);
	int end_f = (int)sz.frames - (int)(border_t1 ? border_t : sPt-1);

	if (params.onlyFrame >=0)
	{
		ori_f = params.onlyFrame;
		end_f = params.onlyFrame + 1;
	}

    // fprintf(stdout,"border_x,border_t: %d,%d\n",border_x,border_t);
    // fprintf(stdout,"ori_*: %d,%d,%d\n",ori_x,ori_y,ori_f);
    // fprintf(stdout,"end_*: %d,%d,%d\n",end_x,end_y,end_f);
    // fprintf(stdout,"step_*: %d,%d,%d\n",stepx,stepy,stepf);

//	ori_f = std::max((int)sz.frames / 2 - (int)sPt - 1, 0);
//	end_f = sz.frames / 2 + 1;

	// Fill processing mask
	for (int f = ori_f, df = 0; f < end_f; f++, df++)
	for (int y = ori_y, dy = 0; y < end_y; y++, dy++)
	for (int x = ori_x, dx = 0; x < end_x; x++, dx++)
	{
		if ( (df % stepf == 0) || (!border_t1 && f == end_f - 1))
		{
			int phasey = (!border_t1 && f == end_f - 1) ? 0 : f/stepf;

			if ( (dy % stepy == phasey % stepy) ||
			     (!border_y1 && y == end_y - 1) ||
			     (!border_y0 && y == ori_y    ) )
			{
				int phasex = (!border_y1 && y == end_y - 1) ? 0 : (phasey + y/stepy);

				if ( (dx % stepx == phasex % stepx) ||
				     (!border_x1 && x == end_x - 1) ||
				     (!border_x0 && x == ori_x    ) )
				{
					mask(x,y,f) = true;
					n_groups++;
				}
			}
		}
	}

	// Used matrices during Bayes' estimate
	const unsigned patch_dim = sPx * sPx * sPt *
      (params.coupleChannels ? sz.channels : 1);
	const unsigned patch_chnls = params.coupleChannels ? 1 : sz.channels;
	const unsigned patch_num = sWx * sWx * sWt;

	// Matrices used for Bayes' estimate
	// std::fprintf(stdout,"patch_num: %d\n",patch_num);
	vector<unsigned> indices(patch_num);
	matWorkspace mat;
	mat.group     .resize(patch_num * patch_dim);
	mat.covMat    .resize(patch_dim * patch_dim);
	mat.center.resize(patch_dim * patch_chnls);

	// Variance captured by the principal components
	Video<float> variance(mask.sz);

	// Total number of groups of similar patches processed
	unsigned group_counter = 0;

	// Allocate output sizes
	if (step1) imBasic.resize(sz); // output of first step
	else       imFinal.resize(sz); // output of second step

    // pick which image to read from for similar patches
    Video<float>* imRead = nullptr;
    if (params.use_imread){ // used in testing only
      imRead = const_cast<Video<float>*>(&imNoisy);
      if (!step1) imRead = const_cast<Video<float>*>(&imBasic);
    }

	// Matrices used for Bayes' estimate
	vector<float> groupNoisy(            patch_num * patch_dim * patch_chnls);
	vector<float> groupBasic(step1 ? 0 : patch_num * patch_dim * patch_chnls);

	// Loop over video
	int remaining_groups = n_groups;
	for (unsigned pt = 0; pt < sz.frames; pt++){
      for (unsigned py = 0; py < sz.height; py++){
        for (unsigned px = 0; px < sz.width ; px++){
        // fprintf(stdout,"iters!: %d,%d,%d\n",pt,py,px);
		if (mask(px,py,pt)) //< Only non-seen patches are processed
		{
            // if (group_counter > 2){
            //   break;
            // }
            // fprintf(stdout,"mask(0,1,2,3): %d,%d,%d,%d\n",
            //         mask(0),mask(1),mask(2),mask(3));
            // fprintf(stdout,"group_counter: %d\n",group_counter);
			group_counter++;
            // fprintf(stdout,"(t,h,w,-): %d,%d,%d,%d\n",pt,py,px,mask(24,0,1));

			const unsigned ij  = sz.index(px,py,pt);
			const unsigned ij3 = sz.index(px,py,pt, 0);
            // fprintf(stdout,"ij,ij3: %d,%d\n",ij,ij3);

			if (params.verbose && (group_counter % 100 == 0))
			{
				int ntiles = crop.ntiles_t * crop.ntiles_x * crop.ntiles_y;
				int part_idx = crop.tile_t * crop.ntiles_x * crop.ntiles_y +
				               crop.tile_y * crop.ntiles_x +
				               crop.tile_x;

				printf("\x1b[%dF[%d,%d,%d] %05.1f\x1b[%dE", ntiles - part_idx,
						crop.tile_x, crop.tile_y, crop.tile_t,
						100.f - (float)remaining_groups/(float)(n_groups)*100.f,
						ntiles - part_idx);

				std::cout << std::flush;
			}

			// Search for similar patches around the reference one
			unsigned nSimP = estimateSimilarPatches(imNoisy, imBasic, fflow, bflow,
                                                    groupNoisy, groupBasic, indices,
                                                    ij3, params, imClean, imRead);
			// If we use the homogeneous area trick
			bool flatPatch = false;
			if (params.flatAreas)
				flatPatch = computeFlatArea(groupNoisy, groupBasic, params, nSimP, sz.channels);
            // fprintf(stdout,"[pre] groupNoisy(0): %2.3f\n",groupNoisy[0]);
            // fprintf(stdout,"[pre] groupNoisy(1): %2.3f\n",groupNoisy[1]);


			// Bayesian estimate
#ifdef FAT_ORIGINAL
			// The Bayesian estimation is skipped with the original Flat Area
			// trick, since the denoising has been done already in the
			// computeFlatArea function.
			if (flatPatch == false)
#endif
              computeBayesEstimate(groupNoisy, groupBasic, mat, params, nSimP, sz.channels, flatPatch);

            // fprintf(stdout,"[post] groupNoisy(0): %2.3f\n",groupNoisy[0]);
            // fprintf(stdout,"[post] groupNoisy(1): %2.3f\n",groupNoisy[1]);

			// Aggregation
			remaining_groups -=
				computeAggregation(step1 ? imBasic : imFinal, weight, mask,
                                   groupNoisy, indices, params, nSimP);
            // fprintf(stdout,"imBasic(0): %2.3f\n",imBasic(0));
            // fprintf(stdout,"imBasic(1): %2.3f\n",imBasic(1));
		}

        }
      }
    }
	// Weighted aggregation
	computeWeightedAggregation(imNoisy, step1 ? imBasic : imFinal, weight);

	if (params.verbose)
	{
		int ntiles = crop.ntiles_t * crop.ntiles_x * crop.ntiles_y;
		int part_idx = crop.tile_t * crop.ntiles_x * crop.ntiles_y +
		               crop.tile_y * crop.ntiles_x +
		               crop.tile_x;

		printf(ANSI_GRN "\x1b[%dF[%d,%d,%d] %05.1f\x1b[%dE" ANSI_RST,
				ntiles - part_idx, crop.tile_x, crop.tile_y,
				crop.tile_t, 100.f, ntiles - part_idx);

		std::cout << std::flush;
	}

	return group_counter;
}

bool compareFirst(
	const std::pair<float, unsigned> &pair1,
	const std::pair<float, unsigned> &pair2)
{
	return pair1.first < pair2.first;
}

unsigned estimateSimilarPatches(
	Video<float> const& imNoisy,
	Video<float> const& imBasic,
	Video<float> const& fflow,
	Video<float> const& bflow,
	std::vector<float> &groupNoisy,
	std::vector<float> &groupBasic,
	std::vector<unsigned> &indices,
	const unsigned pidx,
	const nlbParams &params,
	Video<float> const &imClean,
    Video<float>* imRead)
{
	// Initialization
	bool step1 = params.isFirstStep;
	int sWx   = params.sizeSearchWindow;
	int sWy   = params.sizeSearchWindow;
	const int sWt_f = params.sizeSearchTimeFwd;
	const int sWt_b = params.sizeSearchTimeBwd;
	const int sPx   = params.sizePatch;
	const int sPt   = params.sizePatchTime;
	const VideoSize sz = imNoisy.sz;
	const int dist_chnls = step1 ? 1 : sz.channels;
	bool use_flow = (fflow.sz.width > 0);
    // fprintf(stdout,"sz: %d,%d,%d,%d\n",sz.frames,sz.channels,sz.height,sz.width);

	// Coordinates of center of search box
	unsigned px, py, pt, pc;
	sz.coords(pidx, px, py, pt, pc);
    // std::fprintf(stdout,"(px,py,pt,pc): (%d,%d,%d,%d)\n",px, py, pt, pc);

	// Temporal search range
	int ranget[2];
	int shift_t = std::min(0, (int)pt -  sWt_b)
	            + std::max(0, (int)pt +  sWt_f - (int)sz.frames + sPt);

	ranget[0] = std::max(0, (int)pt - sWt_b - shift_t);
	ranget[1] = std::min((int)sz.frames - sPt, (int)pt +  sWt_f - shift_t);
    // fprintf(stdout,"rt: (%d,%d)\n",ranget[0],ranget[1]);

	// Redefine size of temporal search range
	int sWt = ranget[1] - ranget[0] + 1;
    // fprintf(stdout,"sWt: %d\n",sWt);
    // fprintf(stdout,"pt: %d\n",pt);
    // fprintf(stdout,"(sWt_f,sWt_b): (%d,%d)\n",sWt_f,sWt_b);
    // fprintf(stdout,"(sPt): %d\n",sPt);

	// Allocate vector of patch distances
	std::vector<std::pair<float, unsigned> > distance(sWx * sWy * sWt);

	// Number of patches in search region
	int nsrch = 0;

	// Schedule frames in temporal range
	std::vector<int> srch_ranget;
	srch_ranget.push_back(pt);
	for (int qt = pt+1; qt <= ranget[1]; ++qt) srch_ranget.push_back(qt);
	for (int qt = pt-1; qt >= ranget[0]; --qt) srch_ranget.push_back(qt);

    // print search frames
    // fprintf(stdout,"\nsrch_ranget: ");
    // for (std::vector<int>::iterator it=srch_ranget.begin();it!=srch_ranget.end();it++){
    //   fprintf(stdout,"%d,",*it);
    // }
    // fprintf(stdout,"... [shift = %d]\n",shift_t);

	// Trajectory of search center
	std::vector<int> cx(sWt,0), cy(sWt,0), ct(sWt,0);

	// Search
	for (int ii = 0; ii < sWt; ++ii)
	{
		int qt = srch_ranget[ii]; // video frame number
		int dt = qt - ranget[0]; // search region frame number
		int dir = std::max(-1, std::min(1, qt - (int)pt)); // direction (forward or backwards from pt)
        // std::fprintf(stdout,"(qt,dt,dir): (%d,%d,%d)\n",qt,dt,dir);

		// Include optical flow to new center
		if (dir != 0)
		{
			int cx0 = cx[dt - dir];
			int cy0 = cy[dt - dir];
			int ct0 = ct[dt - dir];
			float cx_f = cx0 + (use_flow ? (dir > 0 ? fflow(cx0,cy0,ct0,0) : bflow(cx0,cy0,ct0,0)) : 0.f);
			float cy_f = cy0 + (use_flow ? (dir > 0 ? fflow(cx0,cy0,ct0,1) : bflow(cx0,cy0,ct0,1)) : 0.f);
            // std::fprintf(stdout,"(cx0,cy0,ct0): (%d,%d,%d)\n",cx0,cy0,ct0);
            // std::fprintf(stdout,"(cx_f,cy_f): (%2.3f,%2.3f)\n",cx_f,cy_f);
			cx[dt] = std::max(0.f, std::min((float)sz.width  - 1, roundf(cx_f)));
			cy[dt] = std::max(0.f, std::min((float)sz.height - 1, roundf(cy_f)));
			ct[dt] = qt;
		}
		else
		{
			cx[dt] = px;
			cy[dt] = py;
			ct[dt] = pt;
		}

		// Spatial search range
		int rangex[2];
		int rangey[2];
        // fprintf(stdout,"rx: (%d,%d) | ry: (%d,%d)\n",
        //         rangex[0],rangex[1],rangey[0],rangey[1]);

		int shift_x = std::min(0, cx[dt] - (sWx-1)/2);
		int shift_y = std::min(0, cy[dt] - (sWy-1)/2);

		shift_x += std::max(0, cx[dt] + (sWx-1)/2 - (int)sz.width  + sPx);
		shift_y += std::max(0, cy[dt] + (sWy-1)/2 - (int)sz.height + sPx);

		rangex[0] = std::max(0, cx[dt] - (sWx-1)/2 - shift_x);
		rangey[0] = std::max(0, cy[dt] - (sWy-1)/2 - shift_y);

		rangex[1] = std::min((int)sz.width  - sPx, cx[dt] + (sWx-1)/2 - shift_x);
		rangey[1] = std::min((int)sz.height - sPx, cy[dt] + (sWy-1)/2 - shift_y);

        // std::fprintf(stdout,"(cx,cy,cz): (%d,%d,%d)\n",cx[dt],cy[dt],ct[dt]);
        // std::fprintf(stdout,"(shift_x,shift_y): (%d,%d)\n",shift_x,shift_y);
        // std::fprintf(stdout,"rangex[0,1] = (%d,%d,%d,%d)\n",rangex[0],rangex[1],shift_x,cx[dt]);
        // std::fprintf(stdout,"rangey[0,1] = (%d,%d,%d,%d)\n",rangey[0],rangey[1],shift_y,cy[dt]);

		// Check if oracle has been provided
		const Video<float> *p_im = imClean.sz.whcf ? &imClean :
		                           (step1 ? &imNoisy : &imBasic);

		// Compute distance between patches in search range
        // fprintf(stdout,"(height,width): (%d,%d)\n",sz.height,sz.width);
        // int pair = sz.index(px + sPx-1, py + sPx-1, pt + sPt-1,0);
        // fprintf(stdout,"(px+hx,py+hy,pt+ht,pair): (%d,%d,%d,%d)\n",px + sPx-1, py + sPx-1, pt + sPt-1,pair);

		for (int qy = rangey[0], dy = 0; qy <= rangey[1]; qy++, dy++)
		for (int qx = rangex[0], dx = 0; qx <= rangex[1]; qx++, dx++)
		{
			// Squared L2 distance
			float dist = 0.f, dif;
			for (int c = 0; c < dist_chnls; c++)
			for (int ht = 0; ht < sPt; ht++)
			for (int hy = 0; hy < sPx; hy++)
			for (int hx = 0; hx < sPx; hx++)
				dist += (dif = (*p_im)(px + hx, py + hy, pt + ht, c)
				             - (*p_im)(qx + hx, qy + hy, qt + ht, c) ) * dif;
            // print for debug
            // int pair_idx = sz.index(qx, qy, qt, 0);
            // std::fprintf(stdout,"pair_idx: %d\n",pair_idx);
            // if (pair_idx == 1411){
            //   std::fprintf(stdout,"(qx,qy,qt): (%d,%d,%d)\n",qx, qy, qt);
            //   std::fprintf(stdout,"dist: %2.2f\n",dist);
            //   for (int ht = 0; ht < sPt; ht++){
            //     for (int hx = 0; hx < sPx; hx++){
            //       for (int hy = 0; hy < sPx; hy++){
            //         for (int c = 0; c < dist_chnls; c++){
            //           std::fprintf(stdout,"%3.1f %d %d %d %d\n",(*p_im)(qx + hx, qy + hy, qt + ht, c),qt+ht,qy+hy,qx+hx,c);
            //         }
            //       }
            //     }
            //   }
            // }

			// Save distance and corresponding patch index
			distance[nsrch++] = std::make_pair(dist, sz.index(qx, qy, qt, 0));
		}
	}

	distance.resize(nsrch);

	// Keep only the nSimilarPatches best similar patches
	unsigned nSimP = std::min(params.nSimilarPatches, (unsigned)distance.size());
	std::partial_sort(distance.begin(), distance.begin() + nSimP,
	                  distance.end(), compareFirst);

    // fprintf(stdout,"nSimilarPatches: %d\n",params.nSimilarPatches);
	if (nSimP < params.nSimilarPatches)
		printf("SR2 [%d,%d,%d] ~ nsim = %d\n", px,py,pt,nSimP);

	// Add more patches if their distance is bellow the threshold
	const float threshold = (params.tau > distance[nSimP - 1].first ?
	                         params.tau : distance[nSimP - 1].first);
	nSimP = 0;
	for (unsigned n = 0; n < distance.size(); n++)
		if (distance[n].first <= threshold)
			indices[nSimP++] = distance[n].second;

	// Save similar patches into 3D groups
	const unsigned w   = sz.width;
	const unsigned wh  = sz.wh;
	const unsigned whc = sz.whc;
    // fprintf(stdout,"coupleChannels: %d\n",params.coupleChannels);
    // fprintf(stdout,"(w,wh,whc): (%d,%d,%d)\n",w,wh,whc);
    // fprintf(stdout,"(sz.channels,sPt,sPx,sPx,nSimP):"
    //         " (%d,%d,%d,%d,%d)\n",sz.channels,sPt,sPx,sPx,nSimP);
    // fprintf(stdout,"(sz.channels,sPt,sPx,sPx,nSimP):"
    //         " (%d,%d,%d,%d,%d)\n",sz.channels,sPt,sPx,sPx,nSimP);
	for (unsigned c = 0, k = 0; c < sz.channels; c++)
	for (unsigned ht = 0; ht < sPt; ht++)
	for (unsigned hy = 0; hy < sPx; hy++)
	for (unsigned hx = 0; hx < sPx; hx++)
	for (unsigned n = 0; n < nSimP; n++, k++)
	{
      // patch around imNoisy[t_i,c_i,y_i,x_i] @ a constant offset of indices[n]
      // problem with this interpt is that we always actually index...
      // t_i = {0,..,sPt}, y_i = {0,..,sPx}, ..
      // so it always indexes the "volume patch" @ the offset of indices[n]
      if (imRead != nullptr){
        groupNoisy[k] = (*imRead)(c * wh + indices[n] + ht * whc + hy * w + hx);
      }else{
        groupNoisy[k] = imNoisy(c * wh + indices[n] + ht * whc + hy * w + hx);
        if (!step1)
          groupBasic[k] = imBasic(c * wh + indices[n] + ht * whc + hy * w + hx);
      }
	}

    // if (pidx == 968){
    //   for(int k = 0; k < 20; ++k){
    //     fprintf(stdout,"groupNoisy[k]: %2.2f | indices[k]: %d | imNoisy[k]: %2.2f\n",
    //             groupNoisy[k],indices[k],imNoisy.data[k]);
    //   }
    // }

	/* 000  pixels from all patches
	 * 001  pixels from all patches
	 * ...
	 * spt,spx,spx pixels from all patches
	 */

	return nSimP;
}

/* Compute variance of a set of patches.
 *
 * patches : array with patches
 * d       : dimensionality of patches
 * n       : number of patches
 * channels: number of channels
 *
 * Returns: the average variance of the set
 */
float computeVariance(
	std::vector<float> const& patches,
	const unsigned d,
	const unsigned n,
	const unsigned channels)
{
	float sigma2 = 0.f;

	for (unsigned c = 0, k = 0; c < channels; c++)
	{
		double sum = 0.f;
		double sum2 = 0.f;

		// Compute the sum and the square sum
		for (unsigned i = 0; i < d * n; i++, k++)
		{
			sum  += patches[k];
			sum2 += patches[k] * patches[k];
		}

		// Sample variance (Bessel's correction)
		sigma2 += (sum2 - sum * sum / (float) (d * n)) / (float) (d * n - 1);
        // fprintf(stdout,"sum2,sum: %2.3f,%2.3f\n",sum2,sum);
	}
    // fprintf(stdout,"d,n: %d,%d\n",d,n);

	return sigma2 / (float) channels;
}

int computeFlatArea(
	std::vector<float> &groupNoisy,
	std::vector<float> const &groupBasic,
	const nlbParams &params,
	const unsigned nSimP,
	const unsigned channels)
{
	const bool s1 = params.isFirstStep;
	const unsigned sP = params.sizePatch * params.sizePatch * params.sizePatchTime;
	const float threshold = params.sigma * params.sigma * params.gamma;

	// Compute the standard deviation of the set of patches
	const float variance = computeVariance(groupNoisy, sP, nSimP, channels);

    // fprintf(stdout,"(var,thresh): %2.3f,%2.3f\n",variance,threshold);

	if (variance < threshold)
	{
#ifdef FAT_ORIGINAL
		// If we are in an homogeneous area, estimate denoised pixels as mean
		std::vector<float>::iterator p_noisy = groupNoisy.begin();
		std::vector<float>::const_iterator p_basic = groupBasic.size()
		                                           ? groupBasic.begin() : p_noisy;
		for (unsigned c = 0; c < channels; c++)
		{
			double mean = 0.f;
			std::vector<float>::const_iterator p_end = p_basic + sP * nSimP;
			for (; p_basic != p_end; ++p_basic) mean += *p_basic;
			mean /= (double)(sP * nSimP);

			p_end = p_noisy + sP * nSimP;
			for (; p_noisy != p_end; ++p_noisy) *p_noisy = mean;
		}
#endif
		return 1;
	}
	else return 0;
}

/* Center patches in group
 *
 * group : group of patches
 * center: center (output)
 * d     : dimensionality of patches
 * n     : number of patches
 */
void centerData(
	std::vector<float> &group,
	std::vector<float> &center,
	const unsigned n,
	const unsigned d)
{
	const float inv = 1.f / (float) n;
	for (unsigned j = 0; j < d; j++) {
		float sum = 0.f;
		for (unsigned i = 0; i < n; i++) {
          sum += group[j * n + i];
		}

		center[j] = sum * inv;

		for (unsigned i = 0; i < n; i++) {
			group[j * n + i] -= center[j];
		}
	}
}

float computeBayesEstimate_FR(
	std::vector<float> &groupNoisy,
	std::vector<float> &groupBasic,
	matWorkspace &mat,
	nlbParams const& params,
	const unsigned nSimP,
	const unsigned chnls,
	const bool flatPatch)
{
	/* Bayesian estimation of patches for a full rank Gaussian model.
	 * Note: this implementation is OBSOLETE. Using full rank matrices
	 * should be avoided (and it is avoided with the default params).
	 */

	// Parameters initialization
	const float diagVal = params.beta * params.sigma * params.sigma;
	const bool s2 = !(params.isFirstStep);
	const unsigned pdim = params.sizePatch * params.sizePatch
	                    * params.sizePatchTime
	                    *(params.coupleChannels ? chnls : 1);
	const unsigned group_chnls = params.coupleChannels ? 1: chnls;

	if (nSimP >= pdim)
	{
		for (unsigned c = 0; c < group_chnls; ++c)
		{
			// in case we need to extract subvectors
			std::vector<float> gBasic_c;
			std::vector<float> gNoisy_c;
			std::vector<float>* gBasic;
			std::vector<float>* gNoisy;
			if (group_chnls == 1)
			{
				// don't need a sub-vector : avoid copy
				gNoisy = &groupNoisy;
				gBasic = s2 ? &groupBasic : gNoisy;
			}
			else
			{
				// we need to extract a subvector: this requires copying
				gNoisy_c = std::vector<float>(groupNoisy.begin() + nSimP*pdim * c   ,
				                              groupNoisy.begin() + nSimP*pdim *(c+1));
				gNoisy = &gNoisy_c;

				if (s2)
				{
					gBasic_c = std::vector<float>(groupBasic.begin() + nSimP*pdim * c   ,
					                              groupBasic.begin() + nSimP*pdim *(c+1));
					gBasic = &gBasic_c;
				}
				else gBasic = gNoisy;
			}

			// Center 3D groups around their center
			if (s2) centerData(*gBasic, mat.center, nSimP, pdim);
			centerData(*gNoisy, mat.center, nSimP, pdim);

			// Compute the covariance matrix of the set of similar patches
			covarianceMatrix(*gBasic, mat.covMat, nSimP, pdim);

			// Bayes' Filtering
			if (s2) for (unsigned k = 0; k < pdim; k++)
				mat.covMat[k * pdim + k] += diagVal;

			// Compute the estimate
			if (inverseMatrix(mat.covMat, pdim) == EXIT_SUCCESS)
			{
				productMatrix(mat.group, mat.covMat, *gNoisy, pdim, pdim, nSimP);
				for (unsigned k = 0; k < pdim * nSimP; k++)
					(*gNoisy)[k] -= diagVal * mat.group[k];
			}

			// Add center
			for (unsigned j = 0, k = 0; j < pdim; j++)
				for (unsigned i = 0; i < nSimP; i++, k++)
					(*gNoisy)[k] += mat.center[j];

			// Copy channel back into vector
			if (group_chnls > 1)
				std::copy(gNoisy->begin(), gNoisy->end(), groupNoisy.begin() + nSimP*pdim*c);
		}
	}

	return 1.f;
}

float computeBayesEstimate_LR(
	std::vector<float> &groupNoisy,
	std::vector<float> &groupBasic,
	matWorkspace &mat,
	nlbParams const& params,
	const unsigned nSimP,
	const unsigned chnls,
	const bool flatPatch)
{
	/* Bayesian estimation using low rank covariance matrix. This functions
	 * computes the eigendecomposition of the covariance matrix using LAPACK.
	 */

	// Parameters initialization
	const bool s2 = !(params.isFirstStep);
	const float sigma2  = params.beta * params.sigma * params.sigma;
	const float sigmab2 = s2 ? params.beta * params.sigmaBasic * params.sigmaBasic : sigma2;
	const float thres   = params.variThres;
	const unsigned r    = params.rank;
	const unsigned pdim = params.sizePatch * params.sizePatch
	                    * params.sizePatchTime
	                    *(params.coupleChannels ? chnls : 1);
	const unsigned group_chnls = params.coupleChannels ? 1 : chnls;
    // fprintf(stdout,"nSimP,pdim,group_chnls: %d,%d,%d\n",nSimP,pdim,group_chnls);
    // fprintf(stdout,"sigma2,sigmab2,variThres: %2.3f,%2.3f,%2.3f\n",
    //         sigma2,sigmab2,thres);
    // fprintf(stdout,"group_chnls,s2: %d,%d\n",group_chnls,s2);

	// Center basic patches with their center
	if (s2) centerData( groupBasic, mat.center, nSimP, pdim * group_chnls);

	// Center 3D groups around their center
	if (s2 && flatPatch)
		// Remove basic's center from Noisy patches
		for (unsigned j = 0, k = 0; j < pdim * group_chnls; j++)
			for (unsigned i = 0; i < nSimP; i++, k++)
				groupNoisy[k] -= mat.center[j];
	else
		// Center noisy patches with their center
		centerData(groupNoisy, mat.center, nSimP, pdim * group_chnls);

	float rank_variance = 0.f;
	if (r > 0)
	{
		for (unsigned c = 0; c < group_chnls; c++)
		{

			std::vector<float> gBasic_c;
			std::vector<float> gNoisy_c;
			std::vector<float>* gBasic;
			std::vector<float>* gNoisy;
			if (group_chnls == 1)
			{
				// don't need a sub-vector : avoid copy
				gNoisy = &groupNoisy;
				gBasic = s2 ? &groupBasic : gNoisy;
			}
			else
			{
				// we need to extract a subvector: this requires copying
				gNoisy_c = std::vector<float>(groupNoisy.begin() + nSimP*pdim * c   ,
				                              groupNoisy.begin() + nSimP*pdim *(c+1));
				gNoisy = &gNoisy_c;

				if (s2)
				{
					gBasic_c = std::vector<float>(groupBasic.begin() +
                                                  nSimP*pdim * c,
					                              groupBasic.begin() +
                                                  nSimP*pdim *(c+1));
					gBasic = &gBasic_c;
				}
				else gBasic = gNoisy;
			}

			// Compute the covariance matrix of the set of similar patches
            // fprintf(stdout,"nSimP: %d, pdim: %d\n",nSimP,pdim);
			covarianceMatrix(*gBasic, mat.covMat, nSimP, pdim);

			// Compute leading eigenvectors
			int info = matrixEigs(mat.covMat, pdim, r,
                                  mat.covEigVals, mat.covEigVecs);

			// Convariance matrix is noisy, either because it has been computed
			// from the noisy patches, or because we model the noise in the basic
			// estimate

			if (sigmab2){
			  modifyEigVals(mat,sigmab2, r, pdim, nSimP, params.var_mode);
			}

			// Compute eigenvalues-based coefficients of Bayes' filter
			/* NOTE: thres is used to control a variance threshold proportional
			 * to the noise. If the eigenvalue is bellow the threshold, the
			 * filter coefficient is set to zero. Otherwise it is set to
			 * the clipped Wiener coefficient.
			 */
            // fprintf(stdout,"thres,sigma2: %2.3f,%2.3f\n",thres,sigma2);
			for (unsigned k = 0; k < r; ++k)
			{
				rank_variance += mat.covEigVals[k];
				mat.covEigVals[k] = (mat.covEigVals[k] > thres * sigma2) ?
				                    1.f / ( 1. + sigma2 / mat.covEigVals[k] ) :
				                    0.f;
			}

			/* NOTE: groupNoisy, if read as a column-major matrix, contains in each
			 * row a patch. Thus, in column-major storage it corresponds to X^T, where
			 * each column of X contains a centered data point.
			 *
			 * We need to compute the noiseless estimage hX as
			 * hX = U * W * U' * X
			 * where U is the matrix with the eigenvectors and W is a diagonal matrix
			 * with the filter coefficients.
			 *
			 * Matrix U is stored (column-major) in mat.covEigVecs. Since we have X^T
			 * we compute
			 * hX' = X' * U * (W * U')
			 */

			// Z' = X'*U
			productMatrix(mat.group,
			              *gNoisy,
			              mat.covEigVecs,
			              nSimP, r, pdim,
			              false, false);

			// U * W
			float *eigVecs = mat.covEigVecs.data();
			for (unsigned k = 0; k < r  ; ++k)
			for (unsigned i = 0; i < pdim; ++i)
				*eigVecs++ *= mat.covEigVals[k];

			// hX' = Z'*(U*W)'
			productMatrix(*gNoisy,
			              mat.group,
			              mat.covEigVecs,
			              nSimP, pdim, r,
			              false, true);

			// Copy channel back into vector
			std::copy(gNoisy->begin(), gNoisy->end(),
                      groupNoisy.begin() + pdim*nSimP*c);
            // fprintf(stdout,"c: %d\n",c);
            // fprintf(stdout,"nSimP: %d\n",nSimP);
            // fprintf(stdout,"pdim: %d\n",pdim);
		}
	}
	else
	{
		// r = 0: set all patches as center
		for (unsigned j = 0, k = 0; j < pdim * group_chnls; j++)
			for (unsigned i = 0; i < nSimP; i++, k++)
				groupNoisy[k] = 0;
	}

	// Add center
	for (unsigned j = 0, k = 0; j < pdim * group_chnls; j++)
		for (unsigned i = 0; i < nSimP; i++, k++)
				groupNoisy[k] += mat.center[j];

	// return variance
	return rank_variance;

}

void modifyEigVals(matWorkspace & mat,
		   float sigmab2, int rank,
		   int pdim, int nSimP, VAR_MODE mode){

  for (int i = 0; i < rank; ++i){
      if (mode == CLIPPED){
        // fprintf(stdout,"[clipped] sigmab2: %2.2f\n",sigmab2);
        mat.covEigVals[i] -= std::min(mat.covEigVals[i], sigmab2);
      }else if(mode == PAUL_VAR){
	float tmp, gamma = (float)pdim/(float)nSimP;
	if (mat.covEigVals[i] > sigmab2 * (tmp = (1 + sqrtf(gamma)))*tmp){
	  tmp = mat.covEigVals[i] - sigmab2 * (1 + gamma);
	  mat.covEigVals[i] = tmp * 0.5
	    * (1. + sqrtf(std::max(0., 1. - 4.*gamma*sigmab2*sigmab2/tmp/tmp)));
	}else{
	  mat.covEigVals[i] = 0;
	}
      }else if(mode == PAUL_SIMPLE){
	float gamma = (float)pdim/(float)nSimP;
	mat.covEigVals[i] -= std::min(mat.covEigVals[i], (1 + gamma) * sigmab2);
      }else if (mode == FAT_OG){
	mat.covEigVals[i] -= sigmab2;
      }
  }
}



float computeBayesEstimate(
	std::vector<float> &groupNoisy,
	std::vector<float> &groupBasic,
	matWorkspace &mat,
	nlbParams const& params,
	const unsigned nSimP,
	const unsigned channels,
	const bool flatPatch)
{
	/* Some images might have a dark noiseless frame (for instance the image
	 * might be the result of a transformation, and a black region correspond to
	 * the pixels that where before the frame before the registration). A group
	 * of patches with 0 variance create problems afterwards. Such groups have to
	 * be avoided. The following code checks if all patches are equal. 
	 *
	 * TODO Check if a subset of the patches are equal, and don't denoise them.
	 */

	// Check if all patches are equal
	const unsigned sPC  = params.sizePatch * params.sizePatch
	                    * params.sizePatchTime * channels;

	for (unsigned j = 0; j < sPC; j++)
	{
		float v = groupNoisy[j * nSimP];
		for (unsigned i = 1; i < nSimP; i++)
			if (v != groupNoisy[j * nSimP + i])
				goto not_equal;
	}

	// All patches are equal ~ do nothing
	return 0.f;

not_equal: // Not all patches are equal ~ denoise

	unsigned pdim = params.sizePatch * params.sizePatch
	              * params.sizePatchTime * (params.coupleChannels ? channels : 1);

	if (nSimP < std::min(pdim, params.rank))
		throw std::runtime_error("VideoNLB::computeBayesEstimate: insufficient patches "
				"for given rank/patch dimension.");

	// Call full or low rank version
	return (params.rank <= pdim)
		? computeBayesEstimate_LR(groupNoisy, groupBasic, mat, params, nSimP, channels, flatPatch)
		: computeBayesEstimate_FR(groupNoisy, groupBasic, mat, params, nSimP, channels, flatPatch);
}

int computeAggregation(
	Video<float> &im,
	Video<float> &weight,
	Video<char>  &mask,
	std::vector<float> const& group,
	std::vector<unsigned> const& indices,
	const nlbParams &params,
	const unsigned nSimP)
{
	// Parameters initializations
	const unsigned chnls = im.sz.channels;
	const unsigned sPx   = params.sizePatch;
	const unsigned sPt   = params.sizePatchTime;
	const unsigned nsPC  = sPx * sPx * sPt * nSimP;

	const unsigned w   = im.sz.width;
	const unsigned h   = im.sz.height;
	const unsigned wh  = im.sz.wh;
	const unsigned whc = im.sz.whc;
    // fprintf(stdout,"psX aggreboost: %d,%d\n",sPx,params.aggreBoost);

	// Aggregate estimates
	int masked = 0;
	for (unsigned n = 0; n < nSimP; n++)
	{
        const unsigned ind = indices[n];
		const unsigned ind1 = (ind / whc) * wh + ind % wh;

		unsigned px0, py0, pt0, pc0;
		im.sz.coords(ind, px0, py0, pt0, pc0);

		// FIXME: this isn't well defined for 3D patches
		if (params.onlyFrame >= 0 && pt0 != params.onlyFrame)
			continue;

		for (unsigned pt = 0; pt < sPt; pt++)
		for (unsigned py = 0; py < sPx; py++)
		for (unsigned px = 0; px < sPx; px++)
		{
			for (unsigned c = 0; c < chnls; c++)
			{
				const unsigned ij = ind  + c * wh;
				im(ij + pt * whc + py * w + px) +=
					group[c * nsPC + (pt * sPx*sPx + py * sPx + px) * nSimP + n];
			}
			weight(ind1 + pt * wh + py * w + px)++;
		}

		// Apply Paste Trick
		unsigned px, py, pt;
		mask.sz.coords(ind1, px, py, pt);
        // fprintf(stdout,"%d,%d,%d,%d,%d\n",masked,mask(ind1),pt,py,px);

		if (mask(ind1)) masked++;
		mask(ind1) = false;

        // if (px == 23 && py == 0 && pt == 0){
        //   fprintf(stdout,"[pre] mask(24,0,1): %d\n",mask(24,0,1));
        // }

		if (params.aggreBoost)
		{
			if ((py >     2*sPx) && mask(ind1 - w)) masked++;
			if ((py < h - 2*sPx) && mask(ind1 + w)) masked++;
			if ((px >     2*sPx) && mask(ind1 - 1)) masked++;
			if ((px < w - 2*sPx) && mask(ind1 + 1)) masked++;

			if (py >     2*sPx) mask(ind1 - w) = false;
			if (py < h - 2*sPx) mask(ind1 + w) = false;
			if (px >     2*sPx) mask(ind1 - 1) = false;
			if (px < w - 2*sPx) mask(ind1 + 1) = false;
		}
        // if (px == 23 && py == 15 && pt == 0){
        //   fprintf(stdout,"[pre] mask(24,0,1): %d\n",mask(24,0,1));
        // }


	}
	// std::fprintf(stdout,"masked: %d\n",masked);

	return masked;
}

void computeWeightedAggregation(
	Video<float> const& im,
	Video<float> & outIm,
	Video<float> const& weight)
{
	for (unsigned f = 0; f < im.sz.frames  ; f++)
	for (unsigned c = 0; c < im.sz.channels; c++)
	for (unsigned y = 0; y < im.sz.height  ; y++)
	for (unsigned x = 0; x < im.sz.width   ; x++)
		if (weight(x,y,f) > 0.f) outIm(x,y,f,c) /= weight(x,y,f);
		else                     outIm(x,y,f,c)  = im(x,y,f,c);
}

}
