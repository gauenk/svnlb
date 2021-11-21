
#
# Copyright (C) Kent Gauen, Purdue University
#

set(PYVNLB_SRC
  cpp/utils/mt19937ar.c
  cpp/utils/VnlbException.cpp
  cpp/utils/VnlbInterrupt.cpp
  cpp/utils/xmalloc.c
  cpp/video_io/iio.c
  cpp/video_io/io.cpp
  cpp/video_io/LibVideoT.cpp
  cpp/flow/tvl1flow_lib.c
  cpp/vnlb/VideoNLBayes.cpp
  cpp/vnlb/LibMatrix.cpp
  cpp/pybind/video_io/read_video_for_flow.cpp
  cpp/pybind/video_io/read_video_for_vnlb.cpp
  cpp/pybind/flow/parser.cpp
  cpp/pybind/flow/tvl1flow.cpp
  cpp/pybind/vnlb/parser.cpp
  cpp/pybind/vnlb/vnlb.cpp
  # cpp/pybind/vnlb/vnlb_timed.cpp
)

set(PYVNLB_HEADERS
  cpp/utils/mt19937ar.h
  cpp/utils/platform_macros.h
  cpp/utils/xmalloc.h
  cpp/utils/VnlbVersion.h
  cpp/utils/VnlbAsserts.h
  cpp/utils/VnlbException.h
  cpp/utils/VnlbInterrupt.h
  cpp/video_io/io.h
  cpp/video_io/iio.h
  cpp/video_io/LibVideoT.hpp
  cpp/flow/defaults.h
  cpp/flow/tvl1flow_lib.h
  cpp/vnlb/vnlb_params.h
  cpp/vnlb/VideoNLBayes.hpp
  cpp/vnlb/LibMatrix.h
  cpp/pybind/interface.h
  cpp/pybind/video_io/interface.h
  cpp/pybind/flow/interface.h
  cpp/pybind/vnlb/interface.h
)


set(PYVNLB_HEADERS ${PYVNLB_HEADERS} PARENT_SCOPE)
add_library(pyvnlb ${PYVNLB_SRC})
add_library(pyvnlb_avx2 ${PYVNLB_SRC})

# Handle `#include <pyvnlb/foo.h>`.
target_include_directories(pyvnlb PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <pyvnlb/foo.h>`.
target_include_directories(pyvnlb_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(pyvnlb PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
set_target_properties(pyvnlb_avx2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(pyvnlb PRIVATE PYVNLB_MAIN_LIB)
  target_compile_definitions(pyvnlb_avx2 PRIVATE PYVNLB_MAIN_LIB)
endif()

target_compile_definitions(pyvnlb PRIVATE FINTEGER=int)
target_compile_definitions(pyvnlb_avx2 PRIVATE FINTEGER=int)

find_package(OpenMP REQUIRED)
target_link_libraries(pyvnlb PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(pyvnlb_avx2 PRIVATE OpenMP::OpenMP_CXX)

find_package( TIFF REQUIRED )
find_package( JPEG REQUIRED )
find_package( PNG  REQUIRED )
target_link_libraries(pyvnlb PRIVATE png tiff)
target_link_libraries(pyvnlb_avx2 PRIVATE png tiff jpeg)

find_package(BLAS REQUIRED)
target_link_libraries(pyvnlb PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(pyvnlb_avx2 PRIVATE ${BLAS_LIBRARIES})

find_package(LAPACKE REQUIRED )
target_link_libraries(pyvnlb PRIVATE ${LAPACK_LIBRARIES})
target_link_libraries(pyvnlb_avx2 PRIVATE ${LAPACK_LIBRARIES})

install(TARGETS pyvnlb
  EXPORT pyvnlb-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(PYVNLB_OPT_LEVEL STREQUAL "avx2")
  install(TARGETS pyvnlb_avx2
    EXPORT pyvnlb-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

foreach(header ${PYVNLB_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyvnlb/${dir}
  )
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/pyvnlb-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/pyvnlb-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/pyvnlb-config.cmake
  COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/pyvnlb-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/pyvnlb-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pyvnlb
)

install(EXPORT pyvnlb-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pyvnlb
)



