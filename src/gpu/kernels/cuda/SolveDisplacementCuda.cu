#include "stdafx.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "solution/Vertex.h"


#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define LHS_MATRIX_INDEX 13            // Position of the LHS matrix in the material config equations
#define EQUATION_ENTRY_SIZE 9 * 27 + 3 // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define BLOCK_SIZE 3                   // Number of threads in one block dimension (total threads per block is BLOCK_SIZE^3)
#define NEUMANN_OFFSET 9 * 27          // Offset to the start of the Neumann stress vector inside an equation block

__global__
void cuda_SolveDisplacement(Vertex* verticesOnGPU, REAL* matConfigEquations, const SolutionDim solutionDimensions, curandState* globalRNGStates) {
    curandState localRNGState = globalRNGStates[threadIdx.x];

    for (int i = 0; i < 100; i++) {
        // choose vertex to update
        int centerCoordX = lroundf(curand_uniform(&localRNGState) * (BLOCK_SIZE-1));
        int centerCoordY = lroundf(curand_uniform(&localRNGState) * (BLOCK_SIZE-1));
        int centerCoordZ = lroundf(curand_uniform(&localRNGState) * (BLOCK_SIZE-1));
        int indexOfCenterVertex = centerCoordZ * BLOCK_SIZE * BLOCK_SIZE + centerCoordY * BLOCK_SIZE + centerCoordX;

        Vertex* globalCenterVertex = &verticesOnGPU[indexOfCenterVertex];

        int equationId = globalCenterVertex->materialConfigId;
        int equationIndex = equationId * (EQUATION_ENTRY_SIZE);

        REAL* matrices = &matConfigEquations[equationIndex];

        REAL rhsVec[3] = { 0,0,0 };
        int localNeighborIndex = 0;
        Vertex zero;

        // Build RHS vector by multiplying each neighbor's displacement with its RHS matrix
        for (char localOffsetZ = 0; localOffsetZ <= 2; localOffsetZ++) {
            for (char localOffsetY = 0; localOffsetY <= 2; localOffsetY++) {
                for (char localOffsetX = 0; localOffsetX <= 2; localOffsetX++) {

                    if (localOffsetZ == 1 && localOffsetY == 1 && localOffsetX == 1) {
                        //This is the center vertex that we're solving for, so skip it
                        continue;
                    }

                    int globalCoordX = centerCoordX + localOffsetX - 1;
                    int globalCoordY = centerCoordY + localOffsetY - 1;
                    int globalCoordZ = centerCoordZ + localOffsetZ - 1;

                    //Local problem size is always 3x3x3 vertices, regardless of solution size
                    localNeighborIndex = localOffsetZ * 9 + localOffsetY * 3 + localOffsetX;

                    const Vertex* neighbor;

                    if (globalCoordX >= 0 && globalCoordX < solutionDimensions.x && 
                        globalCoordY >= 0 && globalCoordY < solutionDimensions.y &&
                        globalCoordZ >= 0 && globalCoordZ < solutionDimensions.z) 
                    {
                        //Neighbor exists
                        neighbor = &verticesOnGPU[globalCoordZ * 9 + globalCoordY * 3 + globalCoordX];
                    }
                    else {
                        //Neighbor is outside the solution space, so the contribution is always zero displacement
                        neighbor = &zero;
                    }

                    REAL nx = neighbor->x;
                    REAL ny = neighbor->y;
                    REAL nz = neighbor->z;

                    // RHS[neighbor] * displacement[neighbor]
                    rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 0) * nx;
                    rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 1) * ny;
                    rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 2) * nz;

                    rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 0) * nx;
                    rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 1) * ny;
                    rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 2) * nz;

                    rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 0) * nx;
                    rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 1) * ny;
                    rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 2) * nz;
                }
            }
        }

        //Move to right side of equation and apply force
        rhsVec[0] = -rhsVec[0] + matrices[NEUMANN_OFFSET];
        rhsVec[1] = -rhsVec[1] + matrices[NEUMANN_OFFSET+1];
        rhsVec[2] = -rhsVec[2] + matrices[NEUMANN_OFFSET+2];

        //rhsVec * LHS^-1
        globalCenterVertex->x =
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 0) * rhsVec[0] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 0) * rhsVec[1] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 0) * rhsVec[2];

        globalCenterVertex->y =
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 1) * rhsVec[0] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 1) * rhsVec[1] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 1) * rhsVec[2];

        globalCenterVertex->z =
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 2) * rhsVec[0] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 2) * rhsVec[1] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 2) * rhsVec[2];

    }

}

__global__
void cuda_init_curand_state(curandState* rngState) {
    int id = threadIdx.x;
    // seed, sequence number, offset, curandState
    curand_init(id, 0, 0, &rngState[id]);
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* vertices, REAL* matConfigEquations, const SolutionDim solutionDims) {

    // setup execution parameters
    //dim3 grid(solutionDims.x / BLOCK_SIZE, solutionDims.y / BLOCK_SIZE, solutionDims.z / BLOCK_SIZE);
    int numThreads = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    // setup curand
    curandState* rngStateOnGPU;
    cudaMalloc(&rngStateOnGPU, sizeof(curandState) * numThreads);
    cuda_init_curand_state << <1, numThreads >> > (rngStateOnGPU);

    cuda_SolveDisplacement << < 1, numThreads >> >(vertices, matConfigEquations, solutionDims, rngStateOnGPU);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda launch failed: %s", cudaGetErrorString(err));
    }
}
