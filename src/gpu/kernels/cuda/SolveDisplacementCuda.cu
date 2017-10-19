#include "stdafx.h"
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "solution/Vertex.h"


#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define LHS_MATRIX_INDEX 13            // Position of the LHS matrix in the material config equations
#define EQUATION_ENTRY_SIZE 9 * 27 + 3 // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define BLOCK_SIZE 4                   // Number of threads in one block dimension (total threads per block is BLOCK_SIZE^3)
#define NEUMANN_OFFSET 9 * 27          // Offset to the start of the Neumann stress vector inside an equation block

__device__
int getGlobalIdx_1D_3D() {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__
int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__
void cuda_SolveDisplacement(Vertex* verticesOnGPU, REAL* matConfigEquations, const SolutionDim solutionDimensions, curandState* globalRNGStates, uint3* blockOrigins) {
    uint3* blockOriginCoord = blockOrigins + blockIdx.x;
    curandState localRNGState = globalRNGStates[getGlobalIdx_1D_3D()];

    for (int i = 0; i < 20; i++) {
        // global coordinate of vertex to update
        int vertexToUpdateX = blockOriginCoord->x + lroundf(curand_uniform(&localRNGState) * (BLOCK_SIZE));
        int vertexToUpdateY = blockOriginCoord->y + lroundf(curand_uniform(&localRNGState) * (BLOCK_SIZE));
        int vertexToUpdateZ = blockOriginCoord->z + lroundf(curand_uniform(&localRNGState) * (BLOCK_SIZE));

        int globalIndexOfCenterVertex = vertexToUpdateZ * solutionDimensions.x * solutionDimensions.y + vertexToUpdateY * solutionDimensions.x + vertexToUpdateX;

        Vertex* globalCenterVertex = &verticesOnGPU[globalIndexOfCenterVertex];

        int equationId = globalCenterVertex->materialConfigId;
        int equationIndex = equationId * (EQUATION_ENTRY_SIZE);

        REAL* matrices = &matConfigEquations[equationIndex];

        REAL rhsVec[3] = { 0,0,0 };
        int localNeighborIndex = 0;
        int globalNeighborIndex = 0;
        Vertex zero;

        // Build RHS vector by multiplying each neighbor's displacement with its RHS matrix
        for (char localOffsetZ = 0; localOffsetZ <= 2; localOffsetZ++) {
            for (char localOffsetY = 0; localOffsetY <= 2; localOffsetY++) {
                for (char localOffsetX = 0; localOffsetX <= 2; localOffsetX++) {

                    if (localOffsetZ == 1 && localOffsetY == 1 && localOffsetX == 1) {
                        //This is the center vertex that we're solving for, so skip it
                        continue;
                    }

                    // Note: globalcoord could be negative which will wrap around to max_int, both would be outside the solution space and caught by the 
                    // if statement below
                    uint3 globalNeighborCoord;
                    globalNeighborCoord.x = vertexToUpdateX + localOffsetX - 1;
                    globalNeighborCoord.y = vertexToUpdateY + localOffsetY - 1;
                    globalNeighborCoord.z = vertexToUpdateZ + localOffsetZ - 1;

                    //Local problem size is always 3x3x3 vertices, regardless of solution size
                    localNeighborIndex = localOffsetZ * 9 + localOffsetY * 3 + localOffsetX;
                    globalNeighborIndex = globalNeighborCoord.z * solutionDimensions.x * solutionDimensions.y + globalNeighborCoord.y * solutionDimensions.x + globalNeighborCoord.x;

                    const Vertex* neighbor;

                    // coords are unsigned but could have wrapped around to max_int, either way they'd be outside the solution space
                    if (globalNeighborCoord.x < solutionDimensions.x &&
                        globalNeighborCoord.y < solutionDimensions.y &&
                        globalNeighborCoord.z < solutionDimensions.z)
                    {
                        //Neighbor exists
                        neighbor = &verticesOnGPU[globalNeighborIndex];
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
        REAL nux = matrices[NEUMANN_OFFSET];
        REAL nuy = matrices[NEUMANN_OFFSET+1];
        REAL nuz = matrices[NEUMANN_OFFSET+2];

        rhsVec[0] = -rhsVec[0] + nux;
        rhsVec[1] = -rhsVec[1] + nuy;
        rhsVec[2] = -rhsVec[2] + nuz;

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
    int id = getGlobalIdx_3D_3D();
    // seed, sequence number, offset, curandState
    curand_init(id, 0, 0, &rngState[id]);
}

__host__
curandState* initializeRNGStates(int numConcurrentBlocks, dim3 threadsPerBlock) {
    int numThreads = numConcurrentBlocks * BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;
    curandState* rngStateOnGPU;
    cudaCheckSuccess(cudaMalloc(&rngStateOnGPU, sizeof(curandState) * numThreads));
    cuda_init_curand_state<<< numConcurrentBlocks, threadsPerBlock >>> (rngStateOnGPU);
    cudaCheckExecution();
    return rngStateOnGPU;
}

__host__
void generateBlockOrigins(uint3* blockOrigins, int numConcurrentBlocks, const SolutionDim solutionDims) {
    std::random_device rd;
    std::mt19937 rng(rd());
    // Choose a vertex as the origin for each block, only choose from vertices that can accommodate a full block around them without
    // any of the vertices inside it being outside the problem space
    std::uniform_int_distribution<int> distX(0, solutionDims.x - BLOCK_SIZE - 1);
    std::uniform_int_distribution<int> distY(0, solutionDims.y - BLOCK_SIZE - 1);
    std::uniform_int_distribution<int> distZ(0, solutionDims.z - BLOCK_SIZE - 1);

    for (int b = 0; b < numConcurrentBlocks; b++) {
        blockOrigins[b].x = distX(rng);
        blockOrigins[b].y = distY(rng);
        blockOrigins[b].z = distZ(rng);
    }
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* vertices, REAL* matConfigEquations, const SolutionDim solutionDims) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // setup execution parameters
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    int numConcurrentBlocks = deviceProperties.multiProcessorCount;

    // setup curand
    curandState* rngStateOnGPU = initializeRNGStates(numConcurrentBlocks, threadsPerBlock);
    cudaDeviceSynchronize();
   
    uint3* blockOrigins;
    cudaCheckSuccess(cudaMallocManaged(&blockOrigins, sizeof(uint3) * numConcurrentBlocks));

    for (int i = 0; i < 100; i++) {
        generateBlockOrigins(blockOrigins, numConcurrentBlocks, solutionDims);
        cuda_SolveDisplacement<<< numConcurrentBlocks, threadsPerBlock >>>(vertices, matConfigEquations, solutionDims, rngStateOnGPU, blockOrigins);
        cudaCheckExecution();
        cudaDeviceSynchronize();
    }

    cudaCheckSuccess(cudaFree(blockOrigins));
    cudaCheckSuccess(cudaFree(rngStateOnGPU));
}


