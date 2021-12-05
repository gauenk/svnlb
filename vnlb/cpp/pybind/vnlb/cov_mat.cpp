
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
#include <vnlb/cpp/vnlb/LibMatrix.h>

#include <vnlb/cpp/pybind/interface.h>
#include <vnlb/cpp/pybind/vnlb/interface.h>

extern "C" {
#include <vnlb/cpp/flow/tvl1flow_lib.h>
#include <vnlb/cpp/video_io/iio.h>
}


void computeCovMatCpp(CovMatParams params){

  // init shapes
  int covSize = params.pdim*params.pdim;
  int eValSize = params.pdim;
  int eVecSize = params.pdim * params.rank;

  // init vectors
  std::vector<float> group(params.gsize);
  std::vector<float> covMat(covSize); // buffer to store the cov matrix
  std::vector<float> covEigVecs; // buffer to store the eigenvecs
  std::vector<float> covEigVals; // buffer to store the eigenvals


  /**************************

      Covariance Matrix

  **************************/

  // exec
  std::memcpy(group.data(),params.groups,params.gsize * sizeof(float));
  covarianceMatrix(group, covMat, params.nSimP, params.pdim);

  // copy back
  float* f_ptr = params.covMat;
  std::memcpy(f_ptr,covMat.data(),covMat.size() * sizeof(float));


  /**************************

         Eigen Stuff

  **************************/

  int info = matrixEigs(covMat, params.pdim, params.rank,
                        covEigVals, covEigVecs);

  // copy back
  // float* f_ptr = params.covMat;
  // std::memcpy(f_ptr,covMat.data(),covMat.size() * sizeof(float));
  f_ptr = params.covEigVals;
  std::memcpy(f_ptr,covEigVals.data(),covEigVals.size() * sizeof(float));
  f_ptr = params.covEigVecs;
  std::memcpy(f_ptr,covEigVecs.data(),covEigVecs.size() * sizeof(float));

}
