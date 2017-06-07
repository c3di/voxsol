
# Module for locating solver dependencies.

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   message(STATUS "CMake thinks this is 64-bit environment")
    set(POSSIBLE_LIB_SUFFIXES Win64 x64 x86_64 lib/Win64 lib/x86_64 lib/x64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   message(STATUS "CMake thinks this is 32-bit environment")
    set(POSSIBLE_LIB_SUFFIXES Win32 x86 lib/Win32 lib/x86)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

SET(SOLVER_DEPENDENCIES_ROOT "dependencies/")

################
# find Ggtest
################

find_path(GTEST_ROOT_DIR
  NAMES lib/x64/gtest.lib
  HINTS ${SOLVER_DEPENDENCIES_ROOT}
  PATH_SUFFIXES gtest-1.8.0
  DOC "google test root directory")

find_path(GTEST_INCLUDE_DIR 
  NAMES gtest
  HINTS ${SOLVER_DEPENDENCIES_ROOT}
  PATH_SUFFIXES gtest-1.8.0/include)

find_library(GTEST_LIBRARY_RELEASE
  NAMES gtest
  HINTS ${GTEST_ROOT_DIR}
  PATH_SUFFIXES ${POSSIBLE_LIB_SUFFIXES})

find_library(GTEST_LIBRARY_DEBUG
  NAMES gtestd
  HINTS ${GTEST_ROOT_DIR}
  PATH_SUFFIXES ${POSSIBLE_LIB_SUFFIXES})  
  
if(GTEST_INCLUDE_DIR)
  message(STATUS "  located google test in ${GTEST_INCLUDE_DIR}")
else()
  message(SEND_ERROR "google test was not found in solver dependencies")
endif()

set(GTEST_BOTH_LIBRARIES debug ${GTEST_LIBRARY_DEBUG} optimized ${GTEST_LIBRARY_RELEASE})
find_package_handle_standard_args(gtest REQUIRED_VARS GTEST_INCLUDE_DIR GTEST_LIBRARY_DEBUG GTEST_LIBRARY_RELEASE)

################
# find CUDA
################

include(FindCUDA)

if (NOT DEFINED CUDA_CUDA_LIBRARY)
    message("This project requires the NVidia Cuda Toolkit v8.0 to be installed")
else()
    include_directories(${CUDA_INCLUDE_DIRS})
    # Enable output of kernel memory useage info during compilation
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_50)
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-Xptxas='-v';--ptxas-options -v)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-Xptxas='-v';--ptxas-options -v)
endif()

################
# find libMMV
################

find_path(LIBMMV_ROOT_DIR
    NAMES include
    HINTS ${SOLVER_DEPENDENCIES_ROOT}
    PATH_SUFFIXES libMMV
)

find_path(LIBMMV_INCLUDE_DIR
    NAMES libmmv
    HINTS ${LIBMMV_ROOT_DIR}
    PATH_SUFFIXES include
)

find_library(LIBMMV_LIBRARY_RELEASE
    NAMES libmmv
    HINTS ${LIBMMV_ROOT_DIR}
    PATH_SUFFIXES lib/x64
)

find_library(LIBMMV_LIBRARY_DEBUG
    NAMES libmmvd
    HINTS ${LIBMMV_ROOT_DIR}
    PATH_SUFFIXES lib/x64
)
