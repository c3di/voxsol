#pragma once
#include <stdafx.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 6                   // Number of threads in one block dimension (total threads per block is BLOCK_SIZE^3)
#define THREADS_PER_BLOCK 216          // Number of total threads per block

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
