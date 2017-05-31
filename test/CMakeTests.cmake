cmake_minimum_required(VERSION 3.4)

INCLUDE(../cmake/CommonFlags)

# find all directories containing project files
set(FILE_TYPES *.c *.cpp *.h *.hpp)
find_directories(DIRS ${TEST_SOURCE_DIR} "${FILE_TYPES}")
# generate source tree
generate_source_tree(HOST_SOURCES "${DIRS}" "${FILE_TYPES}")

set(SOLVER_TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# set include directories
include_directories("${GTEST_INCLUDE_DIR}")

add_executable(${PROJECT_NAME}
			   ${HOST_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
			${GTEST_BOTH_LIBRARIES}
			)
