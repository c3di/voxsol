#pragma once
#include <assert.h>
#include <iostream>

#ifndef NOMINMAX
#define NOMINMAX
#endif

typedef unsigned short ConfigId;

#ifdef USE_DOUBLE_PRECISION

#pragma message("Compiling with DOUBLE precision")
typedef double REAL;
#else
#pragma message("Compiling with SINGLE precision")
typedef float REAL;

#endif

#define asREAL(num) static_cast<REAL>(num)

