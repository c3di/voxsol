#pragma once
#include <cassert>

#ifndef NOMINMAX
#define NOMINMAX
#endif


#ifdef USE_DOUBLE_PRECISION

#pragma message("Compiling with DOUBLE precision")
typedef double REAL;
#else
#pragma message("Compiling with SINGLE precision")
typedef float REAL;

#endif

typedef unsigned short ConfigId;