#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifdef USE_DOUBLE_PRECISION

#pragma message("Compiling with DOUBLE precision")
typedef double REAL;

inline double Real(REAL number) {
    #pragma warning (suppress:  4244 )
    return (double)number;
}

#else

#pragma message("Compiling with SINGLE precision")
typedef float REAL;

inline float Real(REAL number) {
#pragma warning (suppress:  4244 )
    return (float)number;
}

#endif

