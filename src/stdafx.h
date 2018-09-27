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
#define OUTPUT_NUM_FAILED_BLOCKS 1
#define OUTPUT_NAN_DISPLACEMENTS 1
#define OUTPUT_BAD_DISPLACEMENTS 1
#endif

