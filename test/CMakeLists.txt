cmake_minimum_required(VERSION 3.4)

project(stomechUnitTest)
set(TEST_SOURCE_DIR "src/unit")
INCLUDE(CMakeTests.cmake)

project(stomechIntegrationTest)
set(TEST_SOURCE_DIR "src/integration")
INCLUDE(CMakeTests.cmake)
