#include "stdafx.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"


__global__
void cuda_SolveDisplacement(REAL* displacements, ConfigId* matConfigEquationIds, REAL* matConfigEquations) {
    int idx = threadIdx.x*3;
    int equationId = matConfigEquationIds[threadIdx.x];
    int equationIndex = equationId * (27 * 9);

    displacements[idx    ] = equationId;
    displacements[idx + 1] = matConfigEquations[equationIndex + 9*13];
    displacements[idx + 2] = matConfigEquations[equationIndex + 9*13+1];
}

// Is it better to pass raw REAL pointers or create structs for objs like the FragmentSignatures and the displacements (vec3) ?
// Too many arguments? Maybe better to create a KernelParameters struct to store all this stuff in?
extern "C" void cudaLaunchSolveDisplacementKernel(REAL* displacements, ConfigId* matConfigEquationIds, REAL* matConfigEquations, unsigned int numVertices) {

    // setup execution parameters
    dim3 grid(1, 1, 1);
    dim3 threads(64, 1, 1);
    
    cuda_SolveDisplacement <<< grid, threads >>>(displacements, matConfigEquationIds, matConfigEquations);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda launch failed: %s", cudaGetErrorString(err));
    }
}
