
#
# Copyright (C) Kent Gauen, Purdue University
#


########################################
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW) # avoid annoing warnings

project(vnlb
  VERSION 1.0.0
  DESCRIPTION "A python api for the video non-local bayes denoiser."
  HOMEPAGE_URL "https://github.com/gauenk/pyvnlb"
  LANGUAGES CXX C)
include(GNUInstallDirs)

########################################

set(CMAKE_CXX_STANDARD 11)
option(VNLB_ENABLE_PYTHON "Build Python extension." ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE OFF)

########################################

# set(CMAKE_C_FLAGS_RELEASE_INIT "/MT /O2 /Ob2")
# set(CMAKE_CXX_FLAGS_RELEASE_INIT "/MT /O2 /Ob2")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# message(status "cxx release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
# message(status "c release flags: ${CMAKE_C_FLAGS_RELEASE}")


########################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")
include_directories("$ENV{HOME}/local/include/")
set(CMAKE_PREFIX_PATH "$ENV{HOME}/local/")
message(STATUS "Setting CMAKE_PREFIX_PATH as " ${CMAKE_PREFIX_PATH} " for local libraries")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")


########################################

add_subdirectory(vnlb)
add_subdirectory(vnlb/swig)
