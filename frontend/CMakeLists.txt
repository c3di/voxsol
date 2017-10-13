cmake_minimum_required(VERSION 3.10)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

project(stomech_solver CUDA)

# find all directories containing source files
set(FILE_TYPES *.c *.cc *.cu *.cpp *.h *.hpp *.tcc)
find_directories(DIRS src "${FILE_TYPES}")
# generate source tree
generate_source_tree(HOST_SOURCES "${DIRS}" "${FILE_TYPES}")

include_directories(${LIBMMV_INCLUDE_DIR}
                    ${STOMECH_SOLVER_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}
                    ${HOST_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
                      ${LIBMMV_LIBRARY_RELEASE}
					  stomech_solver_lib
)
