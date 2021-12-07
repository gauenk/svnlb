#pragma once

// struct sillMe{
//   sillMe(): sigma(0) {}
//   float sigma;
// };


// allow for use input to change denoising mode
enum VAR_MODE { CLIPPED, PAUL_VAR, PAUL_SIMPLE, FAT_OG };

namespace VideoNLB
{

/* Structures of parameters dedicated to NL-Bayes process
 *
 * NOTES:
 * - Decoupling the color channels of the 3D patches in the 2nd step: Each
 *   channel is considered independently of the others as in the first step.
 *   This makes the 2nd step faster, with results slitghly slower (0.1~0.2 drop
 *   in PSNR)
 * - sigmaBasic is used for the variance estimators based on the theory of
 *   Debashis Paul (PAUL_VARIANCE, PAUL_SIMPLE_VARIANCE)
 * - procStep and aggreBoost are speed ups by reducing the number of processed
 *   patch groups
 */

struct nlbParams {
	float sigma;                // noise standard deviation
	float sigmaBasic;           // std. dev. of remanent noise in the basic estimate
	unsigned sizePatch;         // spatial patch size
	unsigned sizePatchTime;     // temporal patch size
	unsigned nSimilarPatches;   // number of similar patches
	unsigned sizeSearchWindow;  // spatial size of search window (w x w)
	unsigned sizeSearchTimeFwd; // how many forward  frames in search window
	unsigned sizeSearchTimeBwd; // how many backward frames in search window
	bool flatAreas;             // use flat area trick
	float gamma;                // threshold parameter to detect flat areas
	bool coupleChannels;        // joint Gaussian model for all channels
	float variThres;            // variance threshold
	unsigned rank;              // rank of covariance matrix
	float beta;                 // noise multiplier
	float tau;                  // patch distance threshold
	bool isFirstStep;           // which step is it?
	unsigned procStep;          // step used to skip reference patches
	bool aggreBoost;            // if true; patches near denoised patches will be skipped
	int onlyFrame;              // denoise only onlyFrame (-1 means process all frames)
	bool verbose;               // verbose output
	bool testing;               // are we testing?
  VAR_MODE var_mode;
  int nThreads;
  int nParts;

  // to allow for inputs in the swig-python code (hacky)
  bool use_imread;
  bool set_nThreads;
  bool set_nParts;
  bool set_sizePatch;
  bool set_sizePatchTime;
  bool set_nSim;
  bool set_rank;
  bool set_aggreBoost; // added for swig; do we reset the "aggreboost" var?
  bool set_procStep;
};

}
