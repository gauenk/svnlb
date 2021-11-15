
set(VNLB_SRC
  cpp/lib/io.cpp
  cpp/src/pybind/py_res.cpp
  cpp/src/VNLBayes/VideoNLBayes.cpp
  cpp/src/VNLBayes/LibMatrix.cpp
  cpp/src/VidUtils/LibVideoT.cpp
  cpp/src/VidUtils/mt19937ar.c
  cpp/lib/iio/iio.c
  cpp/lib/VnlbException.cpp
  cpp/lib/VnlbInterrupt.cpp
)

set(VNLB_HEADERS
  cpp/lib/VnlbVersion.h
  cpp/lib/platform_macros.h
  cpp/lib/VnlbAsserts.h
  cpp/lib/io.h
  cpp/src/pybind/py_res.h
  cpp/src/VNLBayes/VideoNLBayes.hpp
  cpp/src/VNLBayes/LibMatrix.h
  cpp/src/VidUtils/LibVideoT.hpp
  cpp/src/VidUtils/mt19937ar.h
  cpp/lib/iio/iio.h
  cpp/lib/VnlbException.h
  cpp/lib/VnlbInterrupt.h
)


set(VNLB_HEADERS ${VNLB_HEADERS} PARENT_SCOPE)
add_library(vnlb ${VNLB_SRC})
add_library(vnlb_avx2 ${VNLB_SRC})

# Handle `#include <vnlb/foo.h>`.
target_include_directories(vnlb PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <vnlb/foo.h>`.
target_include_directories(vnlb_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(vnlb PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
set_target_properties(vnlb_avx2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(vnlb PRIVATE VNLB_MAIN_LIB)
  target_compile_definitions(vnlb_avx2 PRIVATE VNLB_MAIN_LIB)
endif()

target_compile_definitions(vnlb PRIVATE FINTEGER=int)
target_compile_definitions(vnlb_avx2 PRIVATE FINTEGER=int)

find_package(OpenMP REQUIRED)
target_link_libraries(vnlb PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(vnlb_avx2 PRIVATE OpenMP::OpenMP_CXX)

find_package( TIFF REQUIRED )
find_package( JPEG REQUIRED )
find_package( PNG  REQUIRED )
target_link_libraries(vnlb PRIVATE png tiff)
target_link_libraries(vnlb_avx2 PRIVATE png tiff jpeg)


#find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(vnlb PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(vnlb_avx2 PRIVATE ${MKL_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(vnlb PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(vnlb_avx2 PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACKE REQUIRED)
  target_link_libraries(vnlb PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(vnlb_avx2 PRIVATE ${LAPACK_LIBRARIES})
endif()

install(TARGETS vnlb
  EXPORT vnlb-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(VNLB_OPT_LEVEL STREQUAL "avx2")
  install(TARGETS vnlb_avx2
    EXPORT vnlb-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

foreach(header ${VNLB_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vnlb/${dir}
  )
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/vnlb-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/vnlb-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/vnlb-config.cmake
  COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/vnlb-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/vnlb-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/vnlb
)

install(EXPORT vnlb-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/vnlb
)



