
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(swigvnlb
  DESCRIPTION "SWIG Python bindings for pyvnlb."
  HOMEPAGE_URL "https://github.com/pariasm/vnlb"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swigpyvnlb source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swigpyvnlb.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigpyvnlb_avx2.swig COPYONLY)

configure_swigpyvnlb(swigpyvnlb.swig)
configure_swigpyvnlb(swigpyvnlb_avx2.swig)

if(TARGET pyvnlb)
  # Manually add headers as extra dependencies of swigpyvnlb.
  set(SWIG_MODULE_swigpyvnlb_EXTRA_DEPS)
  foreach(h ${PYVNLB_HEADERS})
    list(APPEND SWIG_MODULE_swigpyvnlb_EXTRA_DEPS "${pyvnlb_SOURCE_DIR}/pyvnlb/${h}")
    list(APPEND SWIG_MODULE_swigpyvnlb_avx2_EXTRA_DEPS "${pyvnlb_SOURCE_DIR}/pyvnlb/${h}")
  endforeach()
else()
  find_package(pyvnlb REQUIRED)
endif()

swig_add_library(swigpyvnlb
  TYPE SHARED
  LANGUAGE python
  SOURCES swigpyvnlb.swig
)

set_property(SOURCE swigpyvnlb_avx2.swig
  PROPERTY SWIG_MODULE_NAME swigpyvnlb_avx2)
swig_add_library(swigpyvnlb_avx2
  TYPE SHARED
  LANGUAGE python
  SOURCES swigpyvnlb_avx2.swig
)
if(NOT PYVNLB_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(swigpyvnlb_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swigpyvnlb PROPERTIES SUFFIX .so)
  set_target_properties(swigpyvnlb_avx2 PROPERTIES SUFFIX .so)
endif()

if(PYVNLB_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(swigpyvnlb PRIVATE CUDA::cudart)
  target_link_libraries(swigpyvnlb_avx2 PRIVATE CUDA::cudart)
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swigpyvnlb PRIVATE
  pyvnlb
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigpyvnlb_avx2 PRIVATE
  pyvnlb_avx2
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

# Hack so that python_callbacks.h can be included as
# `#include <pyvnlb/python/python_callbacks.h>`.
target_include_directories(swigpyvnlb PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigpyvnlb_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)

find_package(Python REQUIRED
  COMPONENTS Development NumPy
)

add_library(pyvnlb_python_callbacks EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET pyvnlb_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

# Hack so that python_callbacks.h can be included as
# `#include <pyvnlb/python/python_callbacks.h>`.
target_include_directories(pyvnlb_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(pyvnlb_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swigpyvnlb PRIVATE pyvnlb_python_callbacks)
target_link_libraries(swigpyvnlb_avx2 PRIVATE pyvnlb_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)

message(STATUS "The project soure dir " ${PROJECT_SOURCE_DIR} " is where to look")
file(GLOB files "${PROJECT_SOURCE_DIR}/../../pyvnlb/pylib/*")
file(COPY ${files} DESTINATION pylib/)
