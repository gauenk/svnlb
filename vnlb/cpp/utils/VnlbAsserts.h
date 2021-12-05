/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef VNLB_ASSERT_INCLUDED
#define VNLB_ASSERT_INCLUDED

#include <vnlb/cpp/utils/VnlbException.h>
#include <vnlb/cpp/utils/platform_macros.h>
#include <cstdio>
#include <cstdlib>
#include <string>

///
/// Assertions
///

#define VNLB_ASSERT(X)                                  \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Vnlb assertion '%s' failed in %s " \
                    "at %s:%d\n",                        \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__);                           \
            abort();                                     \
        }                                                \
    } while (false)

#define VNLB_ASSERT_MSG(X, MSG)                         \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Vnlb assertion '%s' failed in %s " \
                    "at %s:%d; details: " MSG "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__);                           \
            abort();                                     \
        }                                                \
    } while (false)

#define VNLB_ASSERT_FMT(X, FMT, ...)                    \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Vnlb assertion '%s' failed in %s " \
                    "at %s:%d; details: " FMT "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__,                            \
                    __VA_ARGS__);                        \
            abort();                                     \
        }                                                \
    } while (false)

///
/// Exceptions for returning user errors
///

#define VNLB_THROW_MSG(MSG)                                   \
    do {                                                       \
        throw vnlb::VnlbException(                           \
                MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

#define VNLB_THROW_FMT(FMT, ...)                              \
    do {                                                       \
        std::string __s;                                       \
        int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);   \
        __s.resize(__size + 1);                                \
        snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);       \
        throw vnlb::VnlbException(                           \
                __s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)

///
/// Exceptions thrown upon a conditional failure
///

#define VNLB_THROW_IF_NOT(X)                          \
    do {                                               \
        if (!(X)) {                                    \
            VNLB_THROW_FMT("Error: '%s' failed", #X); \
        }                                              \
    } while (false)

#define VNLB_THROW_IF_NOT_MSG(X, MSG)                       \
    do {                                                     \
        if (!(X)) {                                          \
            VNLB_THROW_FMT("Error: '%s' failed: " MSG, #X); \
        }                                                    \
    } while (false)

#define VNLB_THROW_IF_NOT_FMT(X, FMT, ...)                               \
    do {                                                                  \
        if (!(X)) {                                                       \
            VNLB_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }                                                                 \
    } while (false)

#endif
