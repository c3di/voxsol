#pragma once
#include <assert.h>
#include <iostream>

#ifndef NOMINMAX
#define NOMINMAX
#endif

typedef int ConfigId;
const ConfigId EMPTY_MATERIALS_CONFIG = INT_MAX;

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
//#define OUTPUT_RARE_CONFIGURATIONS_DEBUG 1    // Output local problem configurations that occur <= than this many times in the problem (define 1 for <= 1 times, 2 for <= 2 times etc)
#endif

