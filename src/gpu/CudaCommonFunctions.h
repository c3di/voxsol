#pragma once
#include <stdafx.h>
#include "cuda_runtime.h"

struct SolutionDim {
	unsigned short x;
	unsigned short y;
	unsigned short z;
};

#define cudaCheckSuccess(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA failure: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}