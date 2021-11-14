
// -*- c++ -*-

// Auxiliary index structures, that are used in indexes but that can
// be forward-declared

#ifndef VNLB_INTERRUPT_H
#define VNLB_INTERRUPT_H

#include <stdint.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include <vnlb/cpp/lib/platform_macros.h>

namespace vnlb {


  /***********************************************************
   * Interrupt callback
   ***********************************************************/

  struct VNLB_API InterruptCallback {
    virtual bool want_interrupt() = 0;
    virtual ~InterruptCallback() {}

    // lock that protects concurrent calls to is_interrupted
    static std::mutex lock;

    static std::unique_ptr<InterruptCallback> instance;

    static void clear_instance();

    /** check if:
     * - an interrupt callback is set
     * - the callback returns true
     * if this is the case, then throw an exception. Should not be called
     * from multiple threads.
     */
    static void check();

    /// same as check() but return true if is interrupted instead of
    /// throwing. Can be called from multiple threads.
    static bool is_interrupted();

    /** assuming each iteration takes a certain number of flops, what
     * is a reasonable interval to check for interrupts?
     */
    static size_t get_period_hint(size_t flops);
  };

} // namespace vnlb

#endif
