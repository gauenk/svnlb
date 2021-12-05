
// -*- c++ -*-

#ifndef VNLB_EXCEPTION_INCLUDED
#define VNLB_EXCEPTION_INCLUDED

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace vnlb {

/// Base class for Vnlb exceptions
class VnlbException : public std::exception {
   public:
    explicit VnlbException(const std::string& msg);

    VnlbException(
            const std::string& msg,
            const char* funcName,
            const char* file,
            int line);

    /// from std::exception
    const char* what() const noexcept override;

    std::string msg;
};

/// Handle multiple exceptions from worker threads, throwing an appropriate
/// exception that aggregates the information
/// The pair int is the thread that generated the exception
void handleExceptions(
        std::vector<std::pair<int, std::exception_ptr>>& exceptions);

/** bare-bones unique_ptr
 * this one deletes with delete [] */
template <class T>
struct ScopeDeleter {
    const T* ptr;
    explicit ScopeDeleter(const T* ptr = nullptr) : ptr(ptr) {}
    void release() {
        ptr = nullptr;
    }
    void set(const T* ptr_in) {
        ptr = ptr_in;
    }
    void swap(ScopeDeleter<T>& other) {
        std::swap(ptr, other.ptr);
    }
    ~ScopeDeleter() {
        delete[] ptr;
    }
};

/** same but deletes with the simple delete (least common case) */
template <class T>
struct ScopeDeleter1 {
    const T* ptr;
    explicit ScopeDeleter1(const T* ptr = nullptr) : ptr(ptr) {}
    void release() {
        ptr = nullptr;
    }
    void set(const T* ptr_in) {
        ptr = ptr_in;
    }
    void swap(ScopeDeleter1<T>& other) {
        std::swap(ptr, other.ptr);
    }
    ~ScopeDeleter1() {
        delete ptr;
    }
};

/// make typeids more readable
std::string demangle_cpp_symbol(const char* name);

} // namespace vnlb

#endif
