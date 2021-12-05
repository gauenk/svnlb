

#pragma once

/// Arguments for video denoising
struct VnlbTensors {
VnlbTensors() :
  t(0),h(0),w(0),c(0),
    noisy(nullptr),
    basic(nullptr),
    denoised(nullptr),
    fflow(nullptr),
    bflow(nullptr),
    oracle(nullptr),
    clean(nullptr),
    groupNoisy(nullptr),
    groupBasic(nullptr),
    indices(nullptr),
    use_flow(0),
    use_clean(0),
    use_oracle(0) {}

  // shape
  int t; // nframes
  int h; // height
  int w; // width
  int c; // color

  // floating-point image tensors
  float* noisy;
  float* basic;
  float* denoised;

  float* fflow;
  float* bflow;

  float* oracle;
  float* clean;

  float* groupNoisy;
  float* groupBasic;
  unsigned* indices;

  bool use_flow;
  bool use_clean;
  bool use_oracle;
};
