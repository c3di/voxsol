cmake_minimum_required(VERSION 3.10)

project(voxsolUnitTest CUDA)
set(TEST_SOURCE_DIR "src/unit")
INCLUDE(CMakeTests.cmake)

project(voxsolIntegrationTest CUDA)
set(TEST_SOURCE_DIR "src/integration")
INCLUDE(CMakeTests.cmake)
