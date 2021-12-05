// -*- c++ -*-

#include <algorithm>
#include <cstring>

#include <vnlb/cpp/utils/VnlbInterrupt.h>
#include <vnlb/cpp/utils/VnlbAsserts.h>

namespace vnlb {

  /***********************************************************
   * Interrupt callback
   ***********************************************************/

  std::unique_ptr<InterruptCallback> InterruptCallback::instance;

  std::mutex InterruptCallback::lock;

  void InterruptCallback::clear_instance() {
    delete instance.release();
  }

  void InterruptCallback::check() {
    if (!instance.get()) {
      return;
    }
    if (instance->want_interrupt()) {
      VNLB_THROW_MSG("computation interrupted");
    }
  }

  bool InterruptCallback::is_interrupted() {
    if (!instance.get()) {
      return false;
    }
    std::lock_guard<std::mutex> guard(lock);
    return instance->want_interrupt();
  }

  size_t InterruptCallback::get_period_hint(size_t flops) {
    if (!instance.get()) {
      return 1L << 30; // never check
    }
    // for 10M flops, it is reasonable to check once every 10 iterations
    return std::max((size_t)10 * 10 * 1000 * 1000 / (flops + 1), (size_t)1);
  }

}
