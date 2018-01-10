#include "stdafx.h"
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "solution/Vertex.h"
#include "solution/samplers/BlockSampler.h"


#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define LHS_MATRIX_INDEX 13            // Position of the LHS matrix in the material config equations
#define EQUATION_ENTRY_SIZE 9 * 27 + 3 // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define NEUMANN_OFFSET 9 * 27          // Offset to the start of the Neumann stress vector inside an equation block
#define UPDATES_PER_THREAD 50          // Number of vertices that should be updated stochastically per thread per kernel execution
#define DYN_ADJUSTMENT_MAX 0.001f

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

__device__ void buildRHSVectorForVertex(REAL* rhsVec, Vertex localVertices[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE], const REAL* matrices, short centerCoordX, short centerCoordY, short centerCoordZ) {
    int localNeighborIndex = 0;

    // Build RHS vector by multiplying each neighbor's displacement with its RHS matrix
    for (char localOffsetZ = 0; localOffsetZ <= 2; localOffsetZ++) {
        for (char localOffsetY = 0; localOffsetY <= 2; localOffsetY++) {
            for (char localOffsetX = 0; localOffsetX <= 2; localOffsetX++) {

                if (localOffsetZ == 1 && localOffsetY == 1 && localOffsetX == 1) {
                    //This is the center vertex that we're solving for, so skip it
                    continue;
                }

                unsigned short localNeighborCoordX = centerCoordX + localOffsetX - 1;
                unsigned short localNeighborCoordY = centerCoordY + localOffsetY - 1;
                unsigned short localNeighborCoordZ = centerCoordZ + localOffsetZ - 1;

                //Local problem size is always 3x3x3 vertices, regardless of solution size
                localNeighborIndex = localOffsetZ * 9 + localOffsetY * 3 + localOffsetX;

                Vertex neighbor = localVertices[localNeighborCoordX][localNeighborCoordY][localNeighborCoordZ];

                // RHS[neighbor] * displacement[neighbor]
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 0) * neighbor.x;
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 1) * neighbor.y;
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 2) * neighbor.z;

                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 0) * neighbor.x;
                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 1) * neighbor.y;
                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 2) * neighbor.z;

                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 0) * neighbor.x;
                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 1) * neighbor.y;
                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 2) * neighbor.z;
            }
        }
    }
}

__device__ const REAL* getPointerToMatricesForVertex(Vertex* vertex, const REAL* matConfigEquations) {
    unsigned int equationIndex = static_cast<unsigned int>(vertex->materialConfigId) * (EQUATION_ENTRY_SIZE);
    return &matConfigEquations[equationIndex];
}

// This function is called from inside a conditional, do not place any __syncthreads() in here!
__device__ void updateVerticesStochastically(Vertex localVertices[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE], const REAL* matConfigEquations, curandState localRNGState ) {

    for (int i = 0; i < UPDATES_PER_THREAD; i++) {
        // There's a 1 vertex border around the problem area that shouldn't be updated, so choose something in the middle region
        short offsetX = 1 + lroundf(curand_uniform(&localRNGState) * (WORKING_AREA_SIZE - 1)); 
        short offsetY = 1 + lroundf(curand_uniform(&localRNGState) * (WORKING_AREA_SIZE - 1));
        short offsetZ = 1 + lroundf(curand_uniform(&localRNGState) * (WORKING_AREA_SIZE - 1));

        Vertex* localVertexToUpdate = &localVertices[offsetX][offsetY][offsetZ];
        if (localVertexToUpdate->materialConfigId == static_cast<ConfigId>(0)) {
			// config id 0 should always be the case where the vertex is surrounded by empty cells, therefore not updateable
            continue;
        }
        const REAL* matrices = getPointerToMatricesForVertex(localVertexToUpdate, matConfigEquations);
        REAL rhsVec[3] = { 0,0,0 };
        
        buildRHSVectorForVertex(rhsVec, localVertices, matrices, offsetX, offsetY, offsetZ);

        //Move to right side of equation and apply Neumann stress
        rhsVec[0] = -rhsVec[0] + matrices[NEUMANN_OFFSET];
        rhsVec[1] = -rhsVec[1] + matrices[NEUMANN_OFFSET + 1];
        rhsVec[2] = -rhsVec[2] + matrices[NEUMANN_OFFSET + 2];

        //rhsVec * LHS^-1
        REAL dx = MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 0) * rhsVec[0] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 0) * rhsVec[1] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 0) * rhsVec[2];
        REAL dy = MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 1) * rhsVec[0] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 1) * rhsVec[1] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 1) * rhsVec[2];
        REAL dz = MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 2) * rhsVec[0] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 2) * rhsVec[1] +
            MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 2) * rhsVec[2];

        REAL diff = (abs(localVertexToUpdate->x - dx) + abs(localVertexToUpdate->y - dy) + abs(localVertexToUpdate->z - dz)) * 0.3333333f;
        if (diff > DYN_ADJUSTMENT_MAX) {
	        // Perform dynamical adjustment, discarding any displacement deltas that are larger than the epsilon defined in DYN_ADJUSTMENT_MAX
	        // this is to prevent occasional large errors caused by race conditions. Smaller errors are corrected over time by the stochastic updates
	        // of surrounding vertices
	        printf("Bad adjustment: %f diff for %f,%f,%f \n", diff, rhsVec[0], rhsVec[1], rhsVec[2]);
	        continue;
        }

        localVertexToUpdate->x = dx;
        localVertexToUpdate->y = dy;
        localVertexToUpdate->z = dz;

    }
}

__global__
void cuda_SolveDisplacement(Vertex* verticesOnGPU, REAL* matConfigEquations, const SolutionDim solutionDimensions, curandState* globalRNGStates, const int3* blockOrigins) {
    // Dummy vertex is used for any vertex that lies outside the solution. MatID is designed to cause an exception if one of these vertices is actually worked on
    Vertex dummyVertex;
    dummyVertex.materialConfigId = static_cast<ConfigId>(0);
    int3 blockOriginCoord = blockOrigins[blockIdx.x];
    curandState localRNGState = globalRNGStates[getGlobalIdx_1D_3D()];

    __shared__ Vertex localVertices[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

    short threadVertexX = blockOriginCoord.x + threadIdx.x;
    short threadVertexY = blockOriginCoord.y + threadIdx.y;
    short threadVertexZ = blockOriginCoord.z + threadIdx.z;
    int threadVertexIndex = threadVertexZ * solutionDimensions.x * solutionDimensions.y + threadVertexY * solutionDimensions.x + threadVertexX;

    bool isInsideSolutionSpace = threadVertexX >= 0 && threadVertexX < solutionDimensions.x && 
        threadVertexY >= 0 && threadVertexY < solutionDimensions.y && 
        threadVertexZ >= 0 && threadVertexZ < solutionDimensions.z;

    if (isInsideSolutionSpace) {
        localVertices[threadIdx.x][threadIdx.y][threadIdx.z] = verticesOnGPU[threadVertexIndex];
    } else {
        localVertices[threadIdx.x][threadIdx.y][threadIdx.z] = dummyVertex;
    }

    //__syncthreads must be called outside of any conditional code, but it must be called before going on to ensure shared memory has been initialized by all threads
    __syncthreads();

    float toBeOrNotToBe = curand_uniform(&localRNGState); //kill 50% of threads so we're not randomly updating the same # of vertices as we have threads -> unstable
    if (isInsideSolutionSpace && toBeOrNotToBe > 0.5f) {
        updateVerticesStochastically(localVertices, matConfigEquations, localRNGState);
        verticesOnGPU[threadVertexIndex] = localVertices[threadIdx.x][threadIdx.y][threadIdx.z];
    } else {
         // This thread was responsible for setting one of the border vertices that are not being updated, they're only there to provide input for the neighboring active
         // vertices. After transferring this data to shared memory this thread is finished.
    }
}

__global__
void cuda_init_curand_state(curandState* rngState) {
    int id = getGlobalIdx_3D_3D();
    // seed, sequence number, offset, curandState
    curand_init(clock64(), id, 0, &rngState[id]);
}

__host__
curandState* initializeRNGStates(int numConcurrentBlocks, dim3 threadsPerBlock) {
    int numThreads = numConcurrentBlocks * BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;
    curandState* rngStateOnGPU;
    cudaCheckSuccess(cudaMalloc(&rngStateOnGPU, sizeof(curandState) * numThreads));
    cuda_init_curand_state<<< numConcurrentBlocks, threadsPerBlock >>> (rngStateOnGPU);
    cudaDeviceSynchronize();
    cudaCheckExecution();
    return rngStateOnGPU;
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* vertices, REAL* matConfigEquations, BlockSampler& sampler, const SolutionDim solutionDims) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // setup execution parameters
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount*2; //TODO: Calculate this based on GPU max for # blocks

    // setup curand
    curandState* rngStateOnGPU = initializeRNGStates(maxConcurrentBlocks, threadsPerBlock);

    int3* blockOrigins;
    cudaCheckSuccess(cudaMallocManaged(&blockOrigins, sizeof(int3) * maxConcurrentBlocks));
    
    for (int i = 0; i < 1; i++) {
        int numBlocks = sampler.generateNextBlockOrigins(blockOrigins, maxConcurrentBlocks);
        cuda_SolveDisplacement <<< numBlocks, threadsPerBlock >>>(vertices, matConfigEquations, solutionDims, rngStateOnGPU, blockOrigins);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }

    cudaCheckSuccess(cudaFree(blockOrigins));
    cudaCheckSuccess(cudaFree(rngStateOnGPU));
}


