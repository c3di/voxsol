cmake_minimum_required(VERSION 3.10)

INCLUDE(../cmake/CommonFlags)

# find all directories containing project files
set(FILE_TYPES *.c *.cpp *.h *.hpp)
find_directories(DIRS ${TEST_SOURCE_DIR} "${FILE_TYPES}")
# generate source tree
generate_source_tree(HOST_SOURCES "${DIRS}" "${FILE_TYPES}")

set(SOLVER_TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# silence STL4002 warnings in the gtest-1.8.0 dependency
add_definitions(-D_SILENCE_TR2_SYS_NAMESPACE_DEPRECATION_WARNING -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)

# set include directories
include_directories("${GTEST_INCLUDE_DIR}" 
                     ${LIBMMV_INCLUDE_DIR}
					 ${STOMECH_SOLVER_INCLUDE_DIR}
					 ${TEST_SOURCE_DIR}
)

add_executable(${PROJECT_NAME}
			   ${HOST_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
			${GTEST_BOTH_LIBRARIES}
			${LIBMMV_LIBRARY_RELEASE}
			voxsol_lib
			)

IF (SOLVER_DOUBLE_PRECISION) 
	add_definitions(-DUSE_DOUBLE_PRECISION)
ELSE()
	remove_definitions(-DUSE_DOUBLE_PRECISION)
ENDIF()