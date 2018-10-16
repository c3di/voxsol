#pragma once
#include <assert.h>
#include <iostream>

#ifndef NOMINMAX
#define NOMINMAX
#endif

typedef unsigned int ConfigId;

#ifdef USE_DOUBLE_PRECISION

#pragma message("Compiling with DOUBLE precision")
typedef double REAL;
#else
#pragma message("Compiling with SINGLE precision")
typedef float REAL;

#endif

#define asREAL(num) static_cast<REAL>(num)

//#define DEBUG_OUTPUT_ENABLED 1

#ifdef DEBUG_OUTPUT_ENABLED
//#define OUTPUT_NUM_FAILED_BLOCKS 1    // Number of blocks that were invalidated, eg. for overlapping with another block
//#define OUTPUT_NAN_DISPLACEMENTS 1    // GPU LEVEL: Output a debug message if a NaN displacement is encountered
//#define OUTPUT_BAD_DISPLACEMENTS 1    // GPU LEVEL: Output a debug message if a displacement is discarded due to dynamical adjustment
//#define OUTPUT_NEW_EQUATIONS_DEBUG 1    // Each time a new material configuration is encountered output info about it
#endif

