
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
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(VNLB_ENABLE_PYTHON "Build Python extension." ON)
set(CMAKE_BUILD_TYPE Release)

########################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")
include_directories("$ENV{HOME}/local/include/")
set(CMAKE_PREFIX_PATH "$ENV{HOME}/local/")
message(STATUS "Setting CMAKE_PREFIX_PATH as " ${CMAKE_PREFIX_PATH} " for local libraries")
# find_package( LAPACK REQUIRED )
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb")


########################################
add_subdirectory(vnlb)
add_subdirectory(vnlb/python)
