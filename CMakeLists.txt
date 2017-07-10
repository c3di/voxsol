cmake_minimum_required(VERSION 3.4)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

project(stomech_solver_lib)

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
include(PrecompiledHeaders)
SwitchOnCpp11()

if (MSVC)
	#Compile in static mode to link with gtest dependency
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif(MSVC)

set(SOLVER_DOUBLE_PRECISION FALSE CACHE BOOL "Compile with double precision rather than single precision.")

# find all directories containing source files
set(FILE_TYPES *.c *.cc *.cu *.cpp *.h *.hpp *.tcc)
find_directories(DIRS src "${FILE_TYPES}")
# generate source tree
generate_source_tree(HOST_SOURCES "${DIRS}" "${FILE_TYPES}")

set(STOMECH_SOLVER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(test)
include_directories(${LIBMMV_INCLUDE_DIR}
                    ${STOMECH_SOLVER_INCLUDE_DIR}
)

CUDA_ADD_LIBRARY(${PROJECT_NAME}
                 ${HOST_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
                      ${LIBMMV_LIBRARY_RELEASE}
)

IF (SOLVER_DOUBLE_PRECISION) 
	add_definitions(-DUSE_DOUBLE_PRECISION)
ELSE()
	remove_definitions(-DUSE_DOUBLE_PRECISION)
ENDIF()

# precompiled headers
add_precompiled_header(${PROJECT_NAME} stdafx.h)

add_subdirectory(frontend)
