#include "stdafx.h"
#include <algorithm>
#include <random>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "solution/Vertex.h"
#include "solution/samplers/BlockSampler.h"
#include "gpu/sampling/ResidualVolume.h"
#include "gpu/CudaCommonFunctions.h"
#include "gpu/GPUParameters.h"

#define MATRIX_ENTRY(rhsMatrices, matrixIndex, row, col) __ldg(rhsMatrices + matrixIndex*9 + col*3 + row) //row*27*3 + col*27 + matrixIndex

__constant__ uint3 c_solutionDimensions;
__constant__ uint3 c_residualDimensions;


__device__ __forceinline__ bool isInsideSolution(const int3 coord) {
    return coord.x < c_solutionDimensions.x && coord.y < c_solutionDimensions.y && coord.z < c_solutionDimensions.z &&
           coord.x >= 0 && coord.y >= 0 && coord.z >= 0;
}

__device__ void buildRHSVectorForVertex(
    REAL rhsVec[27][3],
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    const REAL* __restrict__ matConfigEquations,
    const char3& localCenterCoord
) {
    // We want to keep a full warp dedicated to each worker, but we only need enough threads for the 27 neighbors (minus the center vertex)
    const bool threadIsActive = threadIdx.x < 27 && threadIdx.x != CENTER_VERTEX_INDEX;
    unsigned activeThreadMask = __ballot_sync(__activemask(), threadIsActive);

    if (threadIsActive) {
        REAL rhsEntry[3] = { 0,0,0 };

        // Get coords of neighbor that this thread is responsible for, relative to the center vertex, in the 3x3x3 local problem
        const char localNeighborCoordX = (localCenterCoord.x + threadIdx.x % 3) - 1;
        const char localNeighborCoordY = (localCenterCoord.y + (threadIdx.x / 3) % 3) - 1;
        const char localNeighborCoordZ = (localCenterCoord.z + threadIdx.x / 9) - 1;
		
        const REAL nx = localVertices[localNeighborCoordZ][localNeighborCoordY][localNeighborCoordX].x;
        const REAL ny = localVertices[localNeighborCoordZ][localNeighborCoordY][localNeighborCoordX].y;
        const REAL nz = localVertices[localNeighborCoordZ][localNeighborCoordY][localNeighborCoordX].z;

        rhsEntry[0] = MATRIX_ENTRY(matConfigEquations, threadIdx.x, 0, 0) * nx + MATRIX_ENTRY(matConfigEquations, threadIdx.x, 0, 1) * ny + MATRIX_ENTRY(matConfigEquations, threadIdx.x, 0, 2) * nz;
        rhsEntry[1] = MATRIX_ENTRY(matConfigEquations, threadIdx.x, 1, 0) * nx + MATRIX_ENTRY(matConfigEquations, threadIdx.x, 1, 1) * ny + MATRIX_ENTRY(matConfigEquations, threadIdx.x, 1, 2) * nz;
        rhsEntry[2] = MATRIX_ENTRY(matConfigEquations, threadIdx.x, 2, 0) * nx + MATRIX_ENTRY(matConfigEquations, threadIdx.x, 2, 1) * ny + MATRIX_ENTRY(matConfigEquations, threadIdx.x, 2, 2) * nz;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            rhsEntry[0] += __shfl_down_sync(activeThreadMask, rhsEntry[0], offset);
            rhsEntry[1] += __shfl_down_sync(activeThreadMask, rhsEntry[1], offset);
            rhsEntry[2] += __shfl_down_sync(activeThreadMask, rhsEntry[2], offset);
        }

        if (threadIdx.x == 0) {
            // Result of the shuffle reduction is stored in thread 0's variable
            rhsVec[threadIdx.y][0] = rhsEntry[0];
            rhsVec[threadIdx.y][1] = rhsEntry[1];
            rhsVec[threadIdx.y][2] = rhsEntry[2];
        }
    }
}

__device__ void updateVertex(
    Vertex* vertexToUpdate,
    REAL rhsVec[27][3], 
    const REAL* __restrict__ matConfigEquations
) {
    // Choose exactly 3 threads in the same warp to sum up the 3 RHS components and solve the system
	unsigned mask = __ballot_sync(__activemask(), threadIdx.x < 3);
    if (threadIdx.x < 3) {
        const char rhsComponentIndex = threadIdx.x;
        const char workerIndex = threadIdx.y;

        // Move to right side of equation and apply Neumann stress
        rhsVec[workerIndex][rhsComponentIndex] = -rhsVec[workerIndex][rhsComponentIndex] + __ldg(matConfigEquations + NEUMANN_OFFSET + rhsComponentIndex);

        __syncwarp(mask);

        REAL newDisplacement = 0;
        newDisplacement += MATRIX_ENTRY(matConfigEquations, CENTER_VERTEX_INDEX, 0, rhsComponentIndex) * rhsVec[workerIndex][0];
        newDisplacement += MATRIX_ENTRY(matConfigEquations, CENTER_VERTEX_INDEX, 1, rhsComponentIndex) * rhsVec[workerIndex][1];
        newDisplacement += MATRIX_ENTRY(matConfigEquations, CENTER_VERTEX_INDEX, 2, rhsComponentIndex) * rhsVec[workerIndex][2];

        if (rhsComponentIndex == 0) {
            vertexToUpdate->x = newDisplacement;
        }
        if (rhsComponentIndex == 1) {
            vertexToUpdate->y = newDisplacement;
        }
        if (rhsComponentIndex == 2) {
            vertexToUpdate->z = newDisplacement;
        }

    }
}

__device__ void getUpdateCoordForThread(unsigned char subsetIndex, unsigned char vertexIndex, char3* updateCoord) {
    updateCoord->z = (vertexIndex / 9) * 2;
    updateCoord->y = ((vertexIndex / 3) % 3) * 2;
    updateCoord->x = (vertexIndex % 3) * 2;

    //+1 to account for the 1 vertex border around the update region
    updateCoord->x = updateCoord->x + subsetIndex % 2 + 1;
    updateCoord->y = updateCoord->y + (subsetIndex / 2) % 2 + 1;
    updateCoord->z = updateCoord->z + (subsetIndex / 4) % 2 + 1;
}


__device__ void updateVerticesInRegion(
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    const REAL* __restrict__ matConfigEquations
) {
    __shared__ REAL rhsVec[27][3];

    // The local block has a 1-vertex border, so the valid update region actually starts at 1,1,1. This is taken into account below
    char3 localCoord = { 0,0,0 };
    unsigned char subsetIndex = 0;
    unsigned char vertexOffset = 0;

    // There are 9 workers but 27 vertices in each subset, so each subset needs to be divided into 3 sub-subsets
    for (int i = 0; i < UPDATES_PER_VERTEX * 3 * 8; i++) {
        getUpdateCoordForThread(subsetIndex, blockDim.y * vertexOffset + threadIdx.y, &localCoord);

        Vertex* __restrict__ vertexToUpdate = &localVertices[localCoord.z][localCoord.y][localCoord.x];

        if (vertexToUpdate->materialConfigId != EMPTY_MATERIALS_CONFIG) {
            // Config 0 is reserved for vertices surrounded by empty material, these don't need to be processed
            const REAL* __restrict__ matrices = &matConfigEquations[static_cast<int>(vertexToUpdate->materialConfigId) * EQUATION_ENTRY_SIZE];
            buildRHSVectorForVertex(rhsVec, localVertices, matrices, localCoord);
            updateVertex(vertexToUpdate, rhsVec, matrices);
        }

        vertexOffset = vertexOffset + 1;
        if (vertexOffset > 2) {
            vertexOffset = 0;
            subsetIndex = (subsetIndex + 1) % 8;
        }
    }
}

__device__
void copyVerticesFromGlobalToShared(
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    Vertex* verticesOnGPU,
    const int3 blockOriginCoord
) {
    const int blockSizeWithBorder = BLOCK_SIZE + 2;
    const int numThreadsNeeded = blockSizeWithBorder * blockSizeWithBorder; //each thread will copy over a given x,y for all z ("top down")
    int threadIdx_1D = threadIdx.y * 32 + threadIdx.x;

    // Choose the first numThreadsNeeded threads to copy over the vertices
#pragma unroll
    for (unsigned char i = 0; i < 2; i++) {
        if (threadIdx_1D < numThreadsNeeded) {
            const char localCoordX = threadIdx_1D % blockSizeWithBorder;
            const char localCoordY = threadIdx_1D / blockSizeWithBorder;
            char localCoordZ = 0;

            for (int z = 0; z < BLOCK_SIZE + 2; z++) {
                localCoordZ = z;
                const int3 globalCoord = { blockOriginCoord.x + localCoordX - 1, blockOriginCoord.y + localCoordY - 1, blockOriginCoord.z + z - 1 }; //-1 to account for border at both ends
                Vertex* __restrict__ local = &localVertices[localCoordZ][localCoordY][localCoordX];
                local->x = 0;
                local->y = 0;
                local->z = 0;
                local->materialConfigId = 0;

                if (isInsideSolution(globalCoord)) {
                    const int globalIndex = c_solutionDimensions.y*c_solutionDimensions.x*globalCoord.z + c_solutionDimensions.x*globalCoord.y + globalCoord.x;
                    *local = verticesOnGPU[globalIndex];
                }
            }
        }

        // There are less active threads than there are vertices to copy, so we need a second pass for the rest
        threadIdx_1D = blockDim.y * 32 + blockDim.x;
        threadIdx_1D = threadIdx_1D + threadIdx.y * 32 + threadIdx.x;
    }
}

__device__
void copyVerticesFromSharedToGlobalAndUpdateResiduals(
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    Vertex* verticesOnGPU,
    const int3 blockOriginCoord
) {
    const int numThreadsNeeded = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE; //each thread will copy over one vertex in the inner block (without the border)
    int threadIdx_1D = threadIdx.y * 32 + threadIdx.x;

#pragma unroll
    for (unsigned char i = 0; i < 2; i++) {
        if (threadIdx_1D < numThreadsNeeded) {
            const char localCoordZ = 1 + threadIdx_1D / (BLOCK_SIZE * BLOCK_SIZE);
            const char localCoordY = 1 + (threadIdx_1D / BLOCK_SIZE) % BLOCK_SIZE;
            const char localCoordX = 1 + threadIdx_1D % BLOCK_SIZE;

            int3 globalCoord = { 0,0,0 };
            globalCoord.z += localCoordZ + blockOriginCoord.z - 1;
            globalCoord.y += localCoordY + blockOriginCoord.y - 1;
            globalCoord.x += localCoordX + blockOriginCoord.x - 1;

            if (isInsideSolution(globalCoord)) {
                int globalIndex = c_solutionDimensions.y*c_solutionDimensions.x*globalCoord.z + c_solutionDimensions.x*globalCoord.y + globalCoord.x;
                const Vertex* __restrict__ local = &localVertices[localCoordZ][localCoordY][localCoordX];
                Vertex* __restrict__ global = &verticesOnGPU[globalIndex];

                // First set residual to 0 for all updated vertices, then set the outer edge of vertices to the actual residual so future update blocks will
                // be placed near the edges of the current block, where the vertices are no longer in equilibrium.
                /*int residualIndex = (globalCoord.z + 1) / 2 * c_residualDimensions.y * c_residualDimensions.x + (globalCoord.y + 1) / 2 * c_residualDimensions.x + (globalCoord.x + 1) / 2;
                residualVolume[residualIndex] = asREAL(0.0);
                REAL residual = abs(global->x - local->x) + abs(global->y - local->y) + abs(global->z - local->z);
                if (localCoordZ == 1 || localCoordZ == BLOCK_SIZE) {
                    residualVolume[residualIndex] = residual;
                }
                if (localCoordY == 1 || localCoordY == BLOCK_SIZE) {
                    residualVolume[residualIndex] = residual;
                }
                if (localCoordX == 1 || localCoordX == BLOCK_SIZE) {
                    residualVolume[residualIndex] = residual;
                }*/

                *global = *local;

#ifdef OUTPUT_NAN_DISPLACEMENTS
                if (isnan(localVertices[localCoordZ][localCoordY][localCoordX].x)) {
                    printf("NAN encountered for block %i coord %u %u %u \n", blockIdx.x, localCoord.x, localCoord.y, localCoord.z);
                }
#endif
            }
        }

        // There are less active threads than there are vertices to copy, so we need a second pass for the rest
        threadIdx_1D = blockDim.y * 32 + blockDim.x;
        threadIdx_1D = threadIdx_1D + threadIdx.y * 32 + threadIdx.x;
    }
}

__global__
void cuda_SolveDisplacement(
    Vertex* verticesOnGPU,
    const REAL* matConfigEquations,
    const int3* blockOrigins
) {
    const int3 blockOriginCoord = blockOrigins[blockIdx.x];
    if (blockOriginCoord.x >= c_solutionDimensions.x || blockOriginCoord.y >= c_solutionDimensions.y || blockOriginCoord.z >= c_solutionDimensions.z) {
        // Some blocks may have been set to an invalid value during the importance sampling phase if they overlap with some other block, these
        // should not be processed
        return;
    }

    __shared__ Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    copyVerticesFromGlobalToShared(localVertices, verticesOnGPU, blockOriginCoord);

    __syncthreads();

    updateVerticesInRegion(localVertices, matConfigEquations);
    
    __syncthreads();

    copyVerticesFromSharedToGlobalAndUpdateResiduals(localVertices, verticesOnGPU, blockOriginCoord); 

}

__global__
void cuda_invalidateOverlappingBlocks(int3* candidates, const int numberOfCandidates, const unsigned int updateRegionSize) {
    extern __shared__ int3 batch[];
    const int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId >= numberOfCandidates) {
        return;
    }
    int localId = threadIdx.x;
    int3 myCandidate = candidates[globalId];
    batch[localId] = myCandidate;

    __syncthreads();

    // Walk through the candidates toward the left
    while (localId > 0) {
        localId -= 1;
        const int3 leftNeighbor = batch[localId];
        // Check for cube intersection, if any condition is true the two rectangular regions cannot intersect
        bool doesNotIntersect = false;
        doesNotIntersect = doesNotIntersect || leftNeighbor.x >= myCandidate.x + updateRegionSize;
        doesNotIntersect = doesNotIntersect || leftNeighbor.x + updateRegionSize <= myCandidate.x;
        doesNotIntersect = doesNotIntersect || leftNeighbor.z + updateRegionSize <= myCandidate.z;
        doesNotIntersect = doesNotIntersect || leftNeighbor.z >= myCandidate.z + updateRegionSize;
        doesNotIntersect = doesNotIntersect || leftNeighbor.y >= myCandidate.y + updateRegionSize;
        doesNotIntersect = doesNotIntersect || leftNeighbor.y + updateRegionSize <= myCandidate.y;
        if (!doesNotIntersect) {
            // Invalidate this block, it will later be skipped in the update phase since it lies outside the solution by definition of max_uint
            myCandidate.x = INT_MAX;
            myCandidate.y = INT_MAX;
            myCandidate.z = INT_MAX;
            break;
        }
    }

    candidates[globalId] = myCandidate;
}

__host__
extern "C" void cudaLaunchInvalidateOverlappingBlocksKernel(
    int3* candidates,
    const int numCandidatesToFind,
    const int updatePhaseBatchSize
) {

    // Check 'updatePhaseBatchSize' blocks at a time and invalidate any that are overlapping
    // During the update phase the blocks will be processed in batches of this size, and any overlapping blocks in the same batch can cause divergence
    int numBlocks = numCandidatesToFind / updatePhaseBatchSize + (numCandidatesToFind % updatePhaseBatchSize == 0 ? 0 : 1);
    cuda_invalidateOverlappingBlocks << < numBlocks, updatePhaseBatchSize, updatePhaseBatchSize * sizeof(uint3) >> > (candidates, numCandidatesToFind, BLOCK_SIZE);
    cudaDeviceSynchronize();
    cudaCheckExecution();
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernel(
    Vertex* vertices,
    const REAL* matConfigEquations,
    int3* blockOrigins,
    const int numBlockOrigins,
    const uint3 solutionDims
) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // Blocks are divided into warps starting with x, then y, then z
    dim3 threadsPerBlock = { 32, 9, 1 };
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount * 8; //TODO: Calculate this based on GPU max for # blocks
    int numIterations = std::max(numBlockOrigins / maxConcurrentBlocks, 1);

    // Instruct the driver to prefer a larger L1 cache size in hardware where L1 and Shared Memory share the same resource
    // Commented out because it made no difference in performance, but might do in certain situations (eg. bigger block sizes?)
    //cudaCheckSuccess(cudaFuncSetAttribute(cuda_SolveDisplacement, cudaFuncAttributePreferredSharedMemoryCarveout, 33));

    //cudaLaunchInvalidateOverlappingBlocksKernel(blockOrigins, numBlockOrigins, maxConcurrentBlocks);

#ifdef OUTPUT_NUM_FAILED_BLOCKS
    int numFailedBlocks = 0;
    for (int i = 0; i < numBlockOrigins; i++) {
        if (blockOrigins[i].x > solutionDims.x) {
            numFailedBlocks++;
        }
    }
    float percent = (static_cast<float>(numFailedBlocks) / numBlockOrigins) * 100;
    std::cout << numFailedBlocks << " of " << numBlockOrigins << " blocks overlapped (" << percent << "%)" << std::endl;
#endif
    cudaMemcpyToSymbol(c_solutionDimensions, &solutionDims, sizeof(uint3));
    const uint3 residualDims = {(solutionDims.x+1) / 2, (solutionDims.y+1) / 2, (solutionDims.z+1) / 2};
    cudaMemcpyToSymbol(c_residualDimensions, &residualDims, sizeof(uint3));

    // process all blocks in batches of maxConcurrentBlocks
    for (int i = 0; i < numIterations; i++) {
        int3* currentBlockOrigins = &blockOrigins[i * maxConcurrentBlocks];
        int numBlocks = std::min(numBlockOrigins - i*maxConcurrentBlocks, maxConcurrentBlocks);

        cuda_SolveDisplacement <<< numBlocks, threadsPerBlock >>>(vertices, matConfigEquations, currentBlockOrigins);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }

}


