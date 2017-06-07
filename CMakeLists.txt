cmake_minimum_required(VERSION 3.4)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

project(stomech_solver)

set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_USE_RELATIVE_PATHS FALSE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CommonFlags)
include(GetSourceFiles)
include(Cpp11)
include(FindPackageHandleStandardArgs)
include(FindSolverDependencies)
SwitchOnCpp11()

# find all directories containing source files
set(FILE_TYPES *.c *.cc *.cpp *.h *.hpp *.tcc)
find_directories(DIRS src "${FILE_TYPES}")
# generate source tree
generate_source_tree(HOST_SOURCES "${DIRS}" "${FILE_TYPES}")

add_subdirectory(test)
include_directories(${LIBMMV_INCLUDE_DIR})

CUDA_ADD_EXECUTABLE(${PROJECT_NAME}
                    ${HOST_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
                      ${LIBMMV_LIBRARY_RELEASE}
)

