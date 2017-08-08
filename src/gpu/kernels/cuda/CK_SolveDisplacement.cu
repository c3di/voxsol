#include "stdafx.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"


__global__
void cuda_SolveDisplacement(REAL* d_displacements, unsigned short* d_signatureIds, REAL* d_fragmentSignatures) {
    int idx = threadIdx.x*3;
    int sigId = d_signatureIds[threadIdx.x];
    int sigIdx = sigId * 27 * 9;

    d_displacements[idx    ] = sigId;
    d_displacements[idx + 1] = d_fragmentSignatures[sigIdx];
    d_displacements[idx + 2] = d_fragmentSignatures[sigIdx+1];
}

// Is it better to pass raw REAL pointers or create structs for objs like the FragmentSignatures and the displacements (vec3) ?
// Too many arguments? Maybe better to create a KernelParameters struct to store all this stuff in?
extern "C" void CK_SolveDisplacement_launch(REAL* d_displacements, unsigned short* d_signatureIds, REAL* d_fragmentSignatures, unsigned int numVertices) {

    // setup execution parameters
    dim3 grid(1, 1, 1);
    dim3 threads(64, 1, 1);
    
    cuda_SolveDisplacement <<< grid, threads >>>(d_displacements, d_signatureIds, d_fragmentSignatures);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda launch failed: %s", cudaGetErrorString(err));
    }
}
