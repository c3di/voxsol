#include "stdafx.h"
#include <algorithm>
#include <random>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "solution/Vertex.h"
#include "solution/samplers/BlockSampler.h"
#include "gpu/sampling/ResidualVolume.h"
#include "gpu/CudaCommonFunctions.h"
#include "gpu/GPUParameters.h"

#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define NEIGHBOR_INDEX          threadIdx.x
#define RHS_INDEX               threadIdx.y
#define WORKER_INDEX            threadIdx.z

//#define DYN_ADJUSTMENT_MAX 0.01f

__constant__ uint3 c_solutionDimensions;


__device__ bool isInsideSolution(const uint3 coord) {
    return coord.x < c_solutionDimensions.x && coord.y < c_solutionDimensions.y && coord.z < c_solutionDimensions.z;
}

__device__ void buildRHSVectorForVertex(
    REAL rhsVec[NUM_WORKERS][3],
    Vertex localVertices[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2],
    const REAL* matrices,
    const char3& localCenterCoord
) {
    // We want to keep a full warp dedicated to each worker, but we only need enough threads for the 27 neighbors (minus the center vertex)
    const bool threadIsActive = NEIGHBOR_INDEX < 27 && NEIGHBOR_INDEX != CENTER_VERTEX_INDEX;
    unsigned activeThreadMask = __ballot_sync(0xffffffff, threadIsActive);

    if (threadIsActive) {
        REAL rhsEntry = 0;

        // Get coords of neighbor that this thread is responsible for, relative to the center vertex, in the 3x3x3 local problem
        const char localNeighborCoordX = (localCenterCoord.x + NEIGHBOR_INDEX % 3) - 1;
        const char localNeighborCoordY = (localCenterCoord.y + (NEIGHBOR_INDEX / 3) % 3) - 1;
        const char localNeighborCoordZ = (localCenterCoord.z + NEIGHBOR_INDEX / 9) - 1;

        Vertex* neighbor = &localVertices[localNeighborCoordZ][localNeighborCoordY][localNeighborCoordX];

        rhsEntry = MATRIX_ENTRY(matrices, NEIGHBOR_INDEX, RHS_INDEX, 0) * neighbor->x;
        rhsEntry += MATRIX_ENTRY(matrices, NEIGHBOR_INDEX, RHS_INDEX, 1) * neighbor->y;
        rhsEntry += MATRIX_ENTRY(matrices, NEIGHBOR_INDEX, RHS_INDEX, 2) * neighbor->z;

        for (int offset = 16; offset > 0; offset /= 2) {
            rhsEntry += __shfl_down_sync(activeThreadMask, rhsEntry, offset);
        }

        if (NEIGHBOR_INDEX == 0) {
            // Result of the shuffle reduction is stored in thread 0's variable
            rhsVec[WORKER_INDEX][RHS_INDEX] = rhsEntry;
        }
    }
}

__device__ const REAL* getPointerToMatricesForVertexGlobal(Vertex* vertex, const REAL* matConfigEquations) {
    unsigned int equationIndex = static_cast<unsigned int>(vertex->materialConfigId) * (EQUATION_ENTRY_SIZE);
    return &matConfigEquations[equationIndex];
}

__device__ void updateVertex(Vertex* vertexToUpdate, REAL rhsVec[NUM_WORKERS][3], const REAL* matrices) {
    // Choose exactly 3 threads in the same warp to sum up the 3 RHS components and solve the system
    if (RHS_INDEX == 0 && NEIGHBOR_INDEX < 3) {
        const char rhsComponentIndex = threadIdx.x;
        const char workerIndex = WORKER_INDEX;

        // Move to right side of equation and apply Neumann stress
        rhsVec[workerIndex][rhsComponentIndex] = -rhsVec[workerIndex][rhsComponentIndex] + matrices[NEUMANN_OFFSET + rhsComponentIndex];

        __syncwarp();

        REAL newDisplacement = 0;
        newDisplacement += MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 0, rhsComponentIndex) * rhsVec[workerIndex][0];
        newDisplacement += MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 1, rhsComponentIndex) * rhsVec[workerIndex][1];
        newDisplacement += MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 2, rhsComponentIndex) * rhsVec[workerIndex][2];

        if (rhsComponentIndex == 0) {
#ifdef DYN_ADJUSTMENT_MAX
            if (abs(vertexToUpdate->x - newDisplacement) > DYN_ADJUSTMENT_MAX) {
                // Perform dynamical adjustment, discarding any displacement deltas that are larger than the epsilon defined in DYN_ADJUSTMENT_MAX
                // this is to prevent occasional large errors caused by race conditions. Smaller errors are corrected over time by the stochastic updates
                // of surrounding vertices
#ifdef OUTPUT_BAD_DISPLACEMENTS
                printf("Bad adjustment: %f diff for thread %i in block %i and bucket %i\n", newDisplacement, threadIdx.x, blockIdx.x, threadIdx.x / (blockDim.x / 2));
#endif
                return;
            }
#endif
            vertexToUpdate->x = newDisplacement;
        }
        if (rhsComponentIndex == 1) {
#ifdef DYN_ADJUSTMENT_MAX
            if (abs(vertexToUpdate->y - newDisplacement) > DYN_ADJUSTMENT_MAX) {
                // Perform dynamical adjustment, discarding any displacement deltas that are larger than the epsilon defined in DYN_ADJUSTMENT_MAX
                // this is to prevent occasional large errors caused by race conditions. Smaller errors are corrected over time by the stochastic updates
                // of surrounding vertices
#ifdef OUTPUT_BAD_DISPLACEMENTS
                printf("Bad adjustment: %f diff for thread %i in block %i and bucket %i\n", newDisplacement, threadIdx.x, blockIdx.x, threadIdx.x / (blockDim.x / 2));
#endif
                return;
            }
#endif
            vertexToUpdate->y = newDisplacement;
        }
        if (rhsComponentIndex == 2) {
#ifdef DYN_ADJUSTMENT_MAX
            if (abs(vertexToUpdate->z - newDisplacement) > DYN_ADJUSTMENT_MAX) {
                // Perform dynamical adjustment, discarding any displacement deltas that are larger than the epsilon defined in DYN_ADJUSTMENT_MAX
                // this is to prevent occasional large errors caused by race conditions. Smaller errors are corrected over time by the stochastic updates
                // of surrounding vertices
#ifdef OUTPUT_BAD_DISPLACEMENTS
                printf("Bad adjustment: %f diff for thread %i in block %i and bucket %i\n", newDisplacement, threadIdx.x, blockIdx.x, threadIdx.x / (blockDim.x / 2));
#endif
                return;
            }
#endif
            vertexToUpdate->z = newDisplacement;
        }

    }
}


__device__ void updateVerticesStochastically(
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    const REAL* matConfigEquations,
    curandState* localRngState
) {
    __shared__ REAL rhsVec[NUM_WORKERS][3];

    // The local block has a 1-vertex border, so the valid update region actually starts at 1,1,1. This is taken into account below
    char3 localCoord = { 0,0,0 };

    for (int i = 0; i < UPDATES_PER_THREAD; i++) {
        // curand_uniform is 0.0 exclusive, 1.0 inclusive, shift to 1...n+1 for a true uniform distribution and leave it there 
        // since we need to take the 1-vertex border into account anyway
        if (threadIdx.x == 0) {
            localCoord.x = ceilf(curand_uniform(localRngState) * BLOCK_SIZE);
            localCoord.y = ceilf(curand_uniform(localRngState) * BLOCK_SIZE);
            localCoord.z = ceilf(curand_uniform(localRngState) * BLOCK_SIZE);
        }

        localCoord.x = __shfl_sync(0xffffffff, localCoord.x, 0);
        localCoord.y = __shfl_sync(0xffffffff, localCoord.y, 0);
        localCoord.z = __shfl_sync(0xffffffff, localCoord.z, 0);

        Vertex* vertexToUpdate = &localVertices[localCoord.z][localCoord.y][localCoord.x];

        const REAL* matrices = getPointerToMatricesForVertexGlobal(vertexToUpdate, matConfigEquations);

        buildRHSVectorForVertex(rhsVec, localVertices, matrices, localCoord);

        __syncthreads(); // Need to finish reading from shared before we move on to writing, otherwise RHS becomes unstable

        updateVertex(vertexToUpdate, rhsVec, matrices);
    }

}

__device__
void copyVerticesFromGlobalToShared(
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    volatile Vertex* verticesOnGPU,
    const uint3 blockOriginCoord
) {
    const int blockSizeWithBorder = BLOCK_SIZE + 2;
    const int numThreadsNeeded = blockSizeWithBorder * blockSizeWithBorder; //each thread will copy over a given x,y for all z ("top down")
    const int threadIdx_1D = WORKER_INDEX * 32 * 3 + RHS_INDEX * 32 + NEIGHBOR_INDEX;

    // Choose the first numThreadsNeeded threads to copy over the vertices
    if (threadIdx_1D < numThreadsNeeded) {
        const char localCoordX = threadIdx_1D % blockSizeWithBorder;
        const char localCoordY = threadIdx_1D / blockSizeWithBorder;
        char localCoordZ = 0;

        for (int z = 0; z < BLOCK_SIZE + 2; z++) {
            localCoordZ = z;
            const uint3 globalCoord = { blockOriginCoord.x + localCoordX - 1, blockOriginCoord.y + localCoordY - 1, blockOriginCoord.z + z - 1 }; //-1 to account for border at both ends
            Vertex* local = &localVertices[localCoordZ][localCoordY][localCoordX];

            if (isInsideSolution(globalCoord)) {
                const int globalIndex = c_solutionDimensions.y*c_solutionDimensions.x*globalCoord.z + c_solutionDimensions.x*globalCoord.y + globalCoord.x;

                //Turns out it's easier to copy the values manually than to get CUDA to play nice with a volatile struct assignment
                volatile const Vertex* global = &verticesOnGPU[globalIndex];
                local->x = global->x;
                local->y = global->y;
                local->z = global->z;
                local->materialConfigId = global->materialConfigId;
            }
            else {
                local->x = 0;
                local->y = 0;
                local->z = 0;
                local->materialConfigId = 0;
            }
        }
    }
}

__device__
void copyVerticesFromSharedToGlobalAndUpdateResiduals(
    Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    volatile Vertex* verticesOnGPU,
    const uint3 blockOriginCoord,
    REAL* residualVolume
) {
    const int numThreadsNeeded = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE; //each thread will copy over one vertex in the inner block (without the border)
    const int threadIdx_1D = WORKER_INDEX * 32 * 3 + RHS_INDEX * 32 + NEIGHBOR_INDEX;

    if (threadIdx_1D < numThreadsNeeded) {
        const char localCoordZ = 1 + threadIdx_1D / (BLOCK_SIZE * BLOCK_SIZE);
        const char localCoordY = 1 + (threadIdx_1D / BLOCK_SIZE) % BLOCK_SIZE;
        const char localCoordX = 1 + threadIdx_1D % BLOCK_SIZE;

        uint3 globalCoord = { 0,0,0 };
        globalCoord.z += localCoordZ + blockOriginCoord.z - 1;
        globalCoord.y += localCoordY + blockOriginCoord.y - 1;
        globalCoord.x += localCoordX + blockOriginCoord.x - 1;

        if (isInsideSolution(globalCoord)) {
            int globalIndex = c_solutionDimensions.y*c_solutionDimensions.x*globalCoord.z + c_solutionDimensions.x*globalCoord.y + globalCoord.x;
            const Vertex* local = &localVertices[localCoordZ][localCoordY][localCoordX];
            volatile Vertex* global = &verticesOnGPU[globalIndex];

            // First set residual to 0 for all updated vertices, then set the outer edge of vertices to the actual residual so future update blocks will
            // be placed near the edges of the current block, where the vertices are no longer in equilibrium.
            int residualIndex = (globalCoord.z + 1) / 2 * (c_solutionDimensions.y+1) / 2 * (c_solutionDimensions.x+1) / 2 + (globalCoord.y + 1) / 2 * (c_solutionDimensions.x+1) / 2 + (globalCoord.x + 1) / 2;
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
            }

            global->x = local->x;
            global->y = local->y;
            global->z = local->z;

#ifdef OUTPUT_NAN_DISPLACEMENTS
            if (isnan(localVertices[localCoordZ][localCoordY][localCoordX].x)) {
                printf("NAN encountered for block %i coord %u %u %u \n", blockIdx.x, localCoord.x, localCoord.y, localCoord.z);
            }
#endif
        }
    }
}

__global__
void cuda_SolveDisplacement(
    volatile Vertex* verticesOnGPU,
    REAL* matConfigEquations,
    const uint3* blockOrigins,
    curandState* rngState,
    REAL* residualVolume
) {
    const uint3 blockOriginCoord = blockOrigins[blockIdx.x];
    if (blockOriginCoord.x >= c_solutionDimensions.x || blockOriginCoord.y >= c_solutionDimensions.y || blockOriginCoord.z >= c_solutionDimensions.z) {
        // Some blocks may have been set to an invalid value during the importance sampling phase if they overlap with some other block, these
        // should not be processed
        return;
    }

    __shared__ Vertex localVertices[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    copyVerticesFromGlobalToShared(localVertices, verticesOnGPU, blockOriginCoord);
    curandState localRngState = rngState[blockIdx.x * NUM_WORKERS + WORKER_INDEX];

    __syncthreads();

    updateVerticesStochastically(localVertices, matConfigEquations, &localRngState);
    
    __syncthreads();

    copyVerticesFromSharedToGlobalAndUpdateResiduals(localVertices, verticesOnGPU, blockOriginCoord, residualVolume); 

    if (NEIGHBOR_INDEX == 0) {
        // Only thread 0 actually generates values using the RNG state so only this updated copy should be sent back to global
        rngState[blockIdx.x * NUM_WORKERS + WORKER_INDEX] = localRngState;
    }
}

__global__
void cuda_InitCurandState(curandState* rngState) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // seed, sequence number, offset, curandState
    curand_init(43, id, 0, &rngState[id]);
}

__global__
void cuda_invalidateOverlappingBlocks(uint3* candidates, const int numberOfCandidates, const unsigned int updateRegionSize) {
    extern __shared__ uint3 batch[];
    const int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId >= numberOfCandidates) {
        return;
    }
    int localId = threadIdx.x;
    uint3 myCandidate = candidates[globalId];
    batch[localId] = myCandidate;

    __syncthreads();

    // Walk through the candidates toward the left
    while (localId > 0) {
        localId -= 1;
        const uint3 leftNeighbor = batch[localId];
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
            myCandidate.x = UINT_MAX;
            myCandidate.y = UINT_MAX;
            myCandidate.z = UINT_MAX;
            break;
        }
    }

    candidates[globalId] = myCandidate;
}

__host__
extern "C" void cudaLaunchInvalidateOverlappingBlocksKernel(
    uint3* candidates,
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
extern "C" void cudaInitializeRNGStates(curandState** rngStateOnGPU) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // setup execution parameters
    int threadsPerBlock = NUM_WORKERS;
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount * 3; //TODO: Calculate this based on GPU max for # blocks
    int numThreads = maxConcurrentBlocks * threadsPerBlock;

    cudaCheckSuccess(cudaMalloc(rngStateOnGPU, sizeof(curandState) * numThreads));
    cuda_InitCurandState << < maxConcurrentBlocks, threadsPerBlock >> > (*rngStateOnGPU);
    cudaDeviceSynchronize();
    cudaCheckExecution();
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernel(
    volatile Vertex* vertices,
    REAL* matConfigEquations,
    REAL* residualVolume,
    curandState* rngStateOnGPU,
    uint3* blockOrigins,
    const int numBlockOrigins,
    const uint3 solutionDims
) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // Blocks are divided into warps starting with x, then y, then z
    dim3 threadsPerBlock = { 32, 3, NUM_WORKERS };
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount * 3; //TODO: Calculate this based on GPU max for # blocks
    int numIterations = std::max(numBlockOrigins / maxConcurrentBlocks, 1);

    cudaLaunchInvalidateOverlappingBlocksKernel(blockOrigins, numBlockOrigins, maxConcurrentBlocks);

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

    // process all blocks in batches of maxConcurrentBlocks
    for (int i = 0; i < numIterations; i++) {
        uint3* currentBlockOrigins = &blockOrigins[i * maxConcurrentBlocks];
        int numBlocks = std::min(numBlockOrigins - i*maxConcurrentBlocks, maxConcurrentBlocks);

        cuda_SolveDisplacement << < numBlocks, threadsPerBlock >> >(vertices, matConfigEquations, currentBlockOrigins, rngStateOnGPU, residualVolume);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }

}


