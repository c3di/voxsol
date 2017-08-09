#include "stdafx.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"


__global__
void cuda_SolveDisplacement(REAL* d_displacements, ConfigId* d_matConfigEquationIds, REAL* d_matConfigEquations) {
    int idx = threadIdx.x*3;
    int equationId = d_matConfigEquationIds[threadIdx.x];
    int equationIndex = equationId * 27 * 9;

    d_displacements[idx    ] = equationIndex;
    d_displacements[idx + 1] = d_matConfigEquations[equationIndex];
    d_displacements[idx + 2] = d_matConfigEquations[equationIndex +1];
}

// Is it better to pass raw REAL pointers or create structs for objs like the FragmentSignatures and the displacements (vec3) ?
// Too many arguments? Maybe better to create a KernelParameters struct to store all this stuff in?
extern "C" void cudaLaunchSolveDisplacementKernel(REAL* d_displacements, ConfigId* d_matConfigEquationIds, REAL* d_matConfigEquations, unsigned int numVertices) {

    // setup execution parameters
    dim3 grid(1, 1, 1);
    dim3 threads(64, 1, 1);
    
    cuda_SolveDisplacement <<< grid, threads >>>(d_displacements, d_matConfigEquationIds, d_matConfigEquations);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda launch failed: %s", cudaGetErrorString(err));
    }
}
