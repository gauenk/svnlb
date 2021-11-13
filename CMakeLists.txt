
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

project(vnlb
	DESCRIPTION "Python bindings for vnlmb code."
	HOMEPAGE_URL "https://github.com/pariasm/vnlb/"
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

find_package(SWIG REQUIRED COMPONENTS python)
# find_package( OpenCV REQUIRED )
# find_package( OpenCV 4 REQUIRED )
# include_directories([BEFORE] "/home/gauenk/Documents/packages/opencv")
find_package(PythonLibs REQUIRED )

include(${SWIG_USE_FILE})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vnlb/cpp/)

set(CMAKE_SWIG_FLAGS "")
set(SWIG_OUTFILE_DIR ${CMAKE_BINARY_DIR})
set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vnlb/cpp/)
set(PYLIB_DIR ${CMAKE_BINARY_DIR}/../pylib/)

# 
# -- Gaussian Pyramid --
#

# -- cpp --
add_library(GaussianPyramidLib SHARED ${CMAKE_BINARY_DIR}/../cflow/cpp/GaussianPyramid.cpp)

# -- swig --
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/GaussianPyramid.i
  PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/GaussianPyramid.i
  PROPERTY SWIG_MODULE_NAME gpyramid)
set_property(SOURCE gpyramid PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
swig_add_library(gpyramid LANGUAGE python OUTPUT_DIR ${SWIG_OUTFILE_DIR} SOURCES ${CMAKE_BINARY_DIR}/../cflow/python/GaussianPyramid.i ${HEADER_DIR}/GaussianPyramid.h)
target_include_directories(gpyramid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}../cflow/cpp/)
swig_link_libraries(gpyramid GaussianPyramidLib)
swig_link_libraries(gpyramid ${OpenCV_LIBS})

# 
# -- Optical Flow --
#

# -- cpp --
add_library(OpticalFlowLib SHARED ${CMAKE_BINARY_DIR}/../cflow/cpp/OpticalFlow.cpp)

# -- swig --
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/OpticalFlow.i
  PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/OpticalFlow.i
  PROPERTY SWIG_MODULE_NAME cflow)
set_property(SOURCE cflow PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
swig_add_library(cflow LANGUAGE python OUTPUT_DIR ${SWIG_OUTFILE_DIR} SOURCES ${CMAKE_BINARY_DIR}/../cflow/python/OpticalFlow.i ${HEADER_DIR}/OpticalFlow.h)
target_include_directories(cflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}../cflow/cpp/)
swig_link_libraries(cflow OpticalFlowLib)
swig_link_libraries(cflow GaussianPyramidLib)
swig_link_libraries(cflow ${OpenCV_LIBS})


# 
# -- Copy "so" files to install location --
#

macro(copy_to_pylib target file)
  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_BINARY_DIR}/${file}
    ${PYLIB_DIR}/${file}
  )
endmacro()

copy_to_pylib(gpyramid _gpyramid.so)
copy_to_pylib(gpyramid gpyramid.py)
copy_to_pylib(GaussianPyramidLib libGaussianPyramidLib.so)
copy_to_pylib(OpticalFlowLib libOpticalFlowLib.so)
copy_to_pylib(cflow _cflow.so)
copy_to_pylib(cflow cflow.py)



