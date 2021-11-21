


// 	// Compute denoising default parameters
// 	VideoNLB::nlbParams prms1, prms2;
// 	VideoNLB::defaultParameters(prms1, patch_sizex1, patch_sizet1, 1, sigma, noisy.sz, verbose);
// 	VideoNLB::defaultParameters(prms2, patch_sizex2, patch_sizet2, 2, sigma, noisy.sz, verbose);

// 	// Override with command line parameters
// 	if (space_search1 >= 0) VideoNLB::setSizeSearchWindow(prms1, (unsigned)space_search1);
// 	if (space_search2 >= 0) VideoNLB::setSizeSearchWindow(prms2, (unsigned)space_search2);
// 	if (num_patches1  >= 0) VideoNLB::setNSimilarPatches(prms1, (unsigned)num_patches1);
// 	if (num_patches2  >= 0) VideoNLB::setNSimilarPatches(prms2, (unsigned)num_patches2);
// 	if (rank1         >= 0) prms1.rank = rank1;
// 	if (rank2         >= 0) prms2.rank = rank2;
// 	if (thres1        >= 0) prms1.variThres = thres1;
// 	if (thres2        >= 0) prms2.variThres = thres2;
// 	if (beta1         >= 0) prms1.beta = beta1;
// 	if (beta2         >= 0) prms2.beta = beta2;
// 	prms1.flatAreas = flat_area1;
// 	prms2.flatAreas = flat_area2;
// 	prms1.coupleChannels = couple_ch1;
// 	prms2.coupleChannels = couple_ch2;

// 	if (no_paste1) prms1.aggreBoost = false;
// 	if (no_paste2) prms2.aggreBoost = false;

// 	if (patch_step1 >= 0) prms1.procStep = patch_step1;
// 	if (patch_step2 >= 0) prms2.procStep = patch_step2;

// //	prms1.onlyFrame = only_frame - first_frame;
// //	prms2.onlyFrame = only_frame - first_frame;

// 	prms2.sigmaBasic = sigmab;
