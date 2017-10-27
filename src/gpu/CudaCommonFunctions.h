#pragma once
#include <stdafx.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 6                   // Number of threads in one block dimension (total threads per block is BLOCK_SIZE^3)
#define WORKING_AREA_SIZE 4            // Vertices that are 'active', minus the 1 vertex border of reference vertices that aren't updated in the kernel

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

#define cudaCheckExecution() { executionAssert(__FILE__, __LINE__); }
inline void executionAssert(const char *file, int line, bool abort = true) {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution failure: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}