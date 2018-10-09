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

#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define LHS_MATRIX_INDEX        13          // Position of the LHS matrix in the material config equations
#define EQUATION_ENTRY_SIZE     9 * 27 + 3  // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define NEUMANN_OFFSET          9 * 27      // Offset to the start of the Neumann stress vector inside an equation block
#define UPDATES_PER_THREAD      50          // Number of vertices that should be updated stochastically per worker 
#define NUM_WORKERS             6           // Number of vertices per block that are being updated in parallel

#define DYN_ADJUSTMENT_MAX      0.01f

__device__
int getGlobalIdx_1D_3DGlobal() {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__
int getGlobalIdx_3D_3DGlobal() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ bool isInsideSolution(const uint3 coord, const uint3 solutionDimensions) {
    return coord.x < solutionDimensions.x && coord.y < solutionDimensions.y && coord.z < solutionDimensions.z;
}

__device__ void buildRHSVectorForVertex(
    REAL rhsVec[NUM_WORKERS][3][27],
    volatile Vertex localVertices[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2],
    const REAL* matrices,
    const char3& localCenterCoord
) {
    char localNeighborIndex = threadIdx.x;
    char rhsComponentIndex = threadIdx.y;
    char workerIndex = threadIdx.z;

    if (localNeighborIndex > 26) {
        //we run 32 threads instead of 27 to have a full warp dedicated to each direction, but we don't need the 5 extras for the calculations
        return;
    }

    // Get coords of neighbor that this thread is responsible for, relative to the center vertex, in the 3x3x3 local problem
    char3 localNeighborCoord;
    localNeighborCoord.z = (localCenterCoord.z + localNeighborIndex / 9) - 1;
    localNeighborCoord.y = (localCenterCoord.y + (localNeighborIndex /3) % 3) - 1;
    localNeighborCoord.x = (localCenterCoord.x + localNeighborIndex % 3) - 1;

    volatile Vertex* neighbor = &localVertices[localNeighborCoord.x][localNeighborCoord.y][localNeighborCoord.z];
    REAL rhsEntry = 0;

    rhsEntry = MATRIX_ENTRY(matrices, localNeighborIndex, rhsComponentIndex, 0) * neighbor->x;
    rhsEntry += MATRIX_ENTRY(matrices, localNeighborIndex, rhsComponentIndex, 1) * neighbor->y;
    rhsEntry += MATRIX_ENTRY(matrices, localNeighborIndex, rhsComponentIndex, 2) * neighbor->z;

    if (localNeighborIndex == 13) {
        // The center vertex is the one we're solving for, so skip this one
        rhsEntry = 0;
    }

    rhsVec[workerIndex][rhsComponentIndex][localNeighborIndex] = rhsEntry;
}

__device__ const REAL* getPointerToMatricesForVertexGlobal(volatile Vertex* vertex, const REAL* matConfigEquations) {
    unsigned int equationIndex = static_cast<unsigned int>(vertex->materialConfigId) * (EQUATION_ENTRY_SIZE);
    return &matConfigEquations[equationIndex];
}

__device__ void updateVertex(volatile Vertex* vertexToUpdate, REAL rhsVec[NUM_WORKERS][3][27], const REAL* matrices) {
    // Choose exactly 3 threads in the same warp to sum up the 3 RHS components and solve the system
    if (threadIdx.y == 0 && threadIdx.x < 3) {
        REAL rhsValue = 0;
        char rhsComponentIndex = threadIdx.x;
        char workerIndex = threadIdx.z;

        //TODO: use __shfl_down_sync() to perform the reduction?
        for (int i = 0; i < 27; i++) {
            rhsValue += rhsVec[workerIndex][rhsComponentIndex][i];
        }

        // Move to right side of equation and apply Neumann stress
        rhsValue = -rhsValue + matrices[NEUMANN_OFFSET + rhsComponentIndex];

        // Use warp-syncronization to store the values in shared memory so the 3 threads can use them in the next step
        rhsVec[workerIndex][rhsComponentIndex][0] = rhsValue;

        __syncwarp();

        REAL newDisplacement = 0;
        newDisplacement += MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, rhsComponentIndex) * rhsVec[workerIndex][0][0];
        newDisplacement += MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, rhsComponentIndex) * rhsVec[workerIndex][1][0];
        newDisplacement += MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, rhsComponentIndex) * rhsVec[workerIndex][2][0];

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
    volatile Vertex localVertices[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2],
    const REAL* matConfigEquations,
    const unsigned int* localVertexUpdateIndices
) {
    __shared__ REAL rhsVec[NUM_WORKERS][3][27];

    const unsigned int* myVertexListStart = localVertexUpdateIndices + blockIdx.x * blockDim.z * UPDATES_PER_THREAD + threadIdx.z * UPDATES_PER_THREAD;
    for (int i = 0; i < UPDATES_PER_THREAD; i++) {
        const unsigned int vertexIndex = myVertexListStart[i];
        char3 localCoord = { 1,1,1 }; //starts at 1 because 0 is the outer border of fixed vertices, which shouldn't be updated
        localCoord.z += vertexIndex / (BLOCK_SIZE * BLOCK_SIZE);
        localCoord.y += (vertexIndex / BLOCK_SIZE) % BLOCK_SIZE;
        localCoord.x += vertexIndex % BLOCK_SIZE;
        
        volatile Vertex* vertexToUpdate = &localVertices[localCoord.x][localCoord.y][localCoord.z];

        const REAL* matrices = getPointerToMatricesForVertexGlobal(vertexToUpdate, matConfigEquations);

        buildRHSVectorForVertex(rhsVec, localVertices, matrices, localCoord);

        __syncthreads(); // Need to finish reading from shared before we move on to writing, otherwise RHS becomes unstable

        updateVertex(vertexToUpdate, rhsVec, matrices);

        __syncthreads(); // Need to finish writing before moving on to the next update phase, otherwise RHS becomes unstable
    }

}

__device__
void copyVertexFromGlobalToShared(
    volatile Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    volatile Vertex* verticesOnGPU,
    const uint3 blockOriginCoord,
    const uint3 solutionDimensions
) {
    const int blockSizeWithBorder = BLOCK_SIZE + 2;
    const int numThreadsNeeded = blockSizeWithBorder * blockSizeWithBorder; //each thread will copy over a given x,y for all z ("top down")
    const int threadIdx_1D = threadIdx.z * 32 * 3 + threadIdx.y * 32 + threadIdx.x;

    // Choose the first numThreadsNeeded threads to copy over the vertics
    if (threadIdx_1D < numThreadsNeeded) {
        char3 localCoord = { 0,0,0 };
        localCoord.x += threadIdx_1D % blockSizeWithBorder;
        localCoord.y += threadIdx_1D / blockSizeWithBorder;
        localCoord.z = 0;

        for (int z = 0; z < BLOCK_SIZE + 2; z++) {
            localCoord.z = z;
            uint3 globalCoord = { blockOriginCoord.x + localCoord.x - 1, blockOriginCoord.y + localCoord.y - 1, blockOriginCoord.z + z - 1 }; //-1 to account for border at both ends
            volatile Vertex* local = &localVertices[localCoord.x][localCoord.y][localCoord.z];

            if (isInsideSolution(globalCoord, solutionDimensions)) {
                int globalIndex = solutionDimensions.y*solutionDimensions.x*globalCoord.z + solutionDimensions.x*globalCoord.y + globalCoord.x;

                //Turns out it's easier to copy the values manually than to get CUDA to play nice with a volatile struct assignment
                volatile Vertex* global = &verticesOnGPU[globalIndex];
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
void copyVertexFromSharedToGlobal(
    volatile Vertex localVertices[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2],
    volatile Vertex* verticesOnGPU,
    const uint3 blockOriginCoord,
    const uint3 solutionDimensions
) {
    const int numThreadsNeeded = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE; //each thread will copy over one vertex in the inner block (without the border)
    const int threadIdx_1D = threadIdx.z * 32 * 3 + threadIdx.y * 32 + threadIdx.x;

    if (threadIdx_1D < numThreadsNeeded) {
        char3 localCoord = {1,1,1};
        localCoord.z += threadIdx_1D / (BLOCK_SIZE * BLOCK_SIZE);
        localCoord.y += (threadIdx_1D / BLOCK_SIZE) % BLOCK_SIZE;
        localCoord.x += threadIdx_1D % BLOCK_SIZE;

        uint3 globalCoord = { 0,0,0 };
        // Move the checkerboard pattern to the start of the block to be updated
        globalCoord.z += localCoord.z + blockOriginCoord.z - 1;
        globalCoord.y += localCoord.y + blockOriginCoord.y - 1;
        globalCoord.x += localCoord.x + blockOriginCoord.x - 1;

        if (isInsideSolution(globalCoord, solutionDimensions)) {
            int globalIndex = solutionDimensions.y*solutionDimensions.x*globalCoord.z + solutionDimensions.x*globalCoord.y + globalCoord.x;
            volatile Vertex* local = &localVertices[localCoord.x][localCoord.y][localCoord.z];
            volatile Vertex* global = &verticesOnGPU[globalIndex];
            global->x = local->x;
            global->y = local->y;
            global->z = local->z;

#ifdef OUTPUT_NAN_DISPLACEMENTS
            if (isnan(localVertices[localCoord.x][localCoord.y][localCoord.z].x)) {
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
    REAL* residualVolume,
    const uint3 solutionDimensions,
    const uint3* blockOrigins,
    const unsigned int* localVertexUpdateIndices
) {
    const uint3 blockOriginCoord = blockOrigins[blockIdx.x];
    if (blockOriginCoord.x >= solutionDimensions.x || blockOriginCoord.y >= solutionDimensions.y || blockOriginCoord.z >= solutionDimensions.z) {
        // Some blocks may have been set to an invalid value during the importance sampling phase if they overlap with some other block, these
        // should not be processed
        return;
    }

    __shared__ volatile Vertex localVertices[BLOCK_SIZE+2][BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    copyVertexFromGlobalToShared(localVertices, verticesOnGPU, blockOriginCoord, solutionDimensions);

    __syncthreads();

    updateVerticesStochastically(localVertices, matConfigEquations, localVertexUpdateIndices);
    
    __syncthreads();

    copyVertexFromSharedToGlobal(localVertices, verticesOnGPU, blockOriginCoord, solutionDimensions); 
}

__global__
void cuda_permute_vertex_update_list(unsigned int* localVertexUpdateIndices, curandState* rngState) {
    const unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int bucketIndex = blockIdx.x * blockDim.x * UPDATES_PER_THREAD + threadIdx.x * UPDATES_PER_THREAD;
    const unsigned int fullBlockSize = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE - 1;

    curandState localRngState = rngState[threadIndex];
    unsigned int* myListStart = localVertexUpdateIndices + bucketIndex;

    for (int i = 0; i < UPDATES_PER_THREAD; i++) {
        myListStart[i] = curand_uniform(&localRngState) * fullBlockSize;
    }

    rngState[threadIndex] = localRngState;
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
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
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
        uint3 leftNeighbor = batch[localId];
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
            //printf("Invalid block: %i %i %i with %i %i %i\n", myCandidate.x, myCandidate.y, myCandidate.z, leftNeighbor.x, leftNeighbor.y, leftNeighbor.z);
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
    float percent = (static_cast<float>(numFailedBlocks) / numBlockOrigins) * 100);
    std::cout << numFailedBlocks << " of " << numBlockOrigins << " blocks overlapped (" << percent << "%)" << std::endl;
#endif

    unsigned int* localVertexUpdateIndices;
    cudaCheckSuccess(cudaMalloc(&localVertexUpdateIndices, sizeof(unsigned int) * NUM_WORKERS * maxConcurrentBlocks * UPDATES_PER_THREAD));

    // process all blocks in batches of maxConcurrentBlocks
    for (int i = 0; i < numIterations; i++) {
        uint3* currentBlockOrigins = &blockOrigins[i * maxConcurrentBlocks];
        int numBlocks = std::min(numBlockOrigins - i*maxConcurrentBlocks, maxConcurrentBlocks);

        cuda_permute_vertex_update_list << < numBlocks, BLOCK_SIZE >> >(localVertexUpdateIndices, rngStateOnGPU);

        cuda_SolveDisplacement << < numBlocks, threadsPerBlock >> >(vertices, matConfigEquations, residualVolume, solutionDims, currentBlockOrigins, localVertexUpdateIndices);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }

    cudaCheckSuccess(cudaFree(localVertexUpdateIndices));
}


