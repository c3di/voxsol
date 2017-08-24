#include "stdafx.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "solution/Vertex.h"


__global__
void cuda_SolveDisplacement(Vertex* verticesOnGPU, REAL* matConfigEquations) {
    int idx = threadIdx.x*3;
    Vertex* vertex = &verticesOnGPU[threadIdx.x];
    int equationId = vertex->materialConfigId;
    int equationIndex = equationId * (27 * 9 + 3);

    vertex->x = matConfigEquations[equationIndex + 9 * 13];
    vertex->y = matConfigEquations[equationIndex + 9*13+4];
    vertex->z = matConfigEquations[equationIndex + 9*13+5];
}

extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* vertices, REAL* matConfigEquations, unsigned int numVertices) {

    // setup execution parameters
    dim3 grid(1, 1, 1);
    dim3 threads(64, 1, 1);
    
    cuda_SolveDisplacement <<< grid, threads >>>(vertices, matConfigEquations);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda launch failed: %s", cudaGetErrorString(err));
    }
}
