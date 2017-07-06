#include "stdafx.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define cudaCheckSuccess(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void cuda_ComputeLHSMatrices(REAL* integrals_linear ) {

}
