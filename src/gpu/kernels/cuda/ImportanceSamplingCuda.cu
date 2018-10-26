#include "stdafx.h"
#include <algorithm>
#include <random>
#include <limits>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "gpu/sampling/ResidualVolume.h"
#include "solution/Vertex.h"
#include "gpu/GPUParameters.h"

__device__ REAL cuda_getPyramidValue(const REAL* importancePyramid, const LevelStats levelStats, int x, int y, int z) {
    int globalIndex = levelStats.startIndex + z * levelStats.sizeX * levelStats.sizeY + y * levelStats.sizeX + x;
    return importancePyramid[globalIndex];
}

__device__
void cuda_traversePyramid(const REAL* importancePyramid, const LevelStats* levelStats, int3* position, float remainderResidual, int level) {
    REAL splitValue = asREAL(0.0);
    for (int z = 0; z < 2; z++)
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++) {
                float lastResidual = splitValue;
                splitValue += cuda_getPyramidValue(importancePyramid, levelStats[level], position->x + x, position->y + y, position->z + z);
                if (remainderResidual < splitValue) {
                    position->x = (position->x + x) * 2;
                    position->y = (position->y + y) * 2;
                    position->z = (position->z + z) * 2;

                    if (level == 0) {
                        // Note: level 0 is fullres/2, so even when we've reached level 0 we still multiply by 2 once more ^
                        return;
                    }

                    level -= 1;
                    remainderResidual -= lastResidual;

                    return cuda_traversePyramid(importancePyramid, levelStats, position, remainderResidual, level);
                }
            }
    //printf("Pyramid traversal failed to find a compatible vertex \n");
}

__global__ 
void cuda_selectImportanceSamplingCandidates(uint3* candidates, const REAL* importancePyramid, const LevelStats* levelStats, curandState* rngState, const int topLevel, const unsigned int updateRegionSize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localRNGState = rngState[id];
    REAL totalResidual = importancePyramid[levelStats[topLevel].startIndex];
    REAL diceRoll = curand_uniform(&localRNGState) * totalResidual;
    int3 position = make_int3(0, 0, 0);

    cuda_traversePyramid(importancePyramid, levelStats, &position, diceRoll, topLevel);

    // if the block origin would cause most of the block to be outside the solution space, push it inward so that its outer edge is on the outer edge of the solution space
    position.x -= max(position.x + (int)updateRegionSize - (int)levelStats[0].sizeX*2, 0); //*2 because level 0 is already half the size of the solution space
    position.y -= max(position.y + (int)updateRegionSize - (int)levelStats[0].sizeY*2, 0);
    position.z -= max(position.z + (int)updateRegionSize - (int)levelStats[0].sizeZ*2, 0);

    // convert to unsigned int, clip negative coordinates to 0
    uint3 pos = make_uint3(max(position.x, 0), max(position.y, 0), max(position.z, 0));
    candidates[id] = pos;
}

__global__
void cuda_init_curand_statePyramid(curandState* rngState) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // seed, sequence number, offset, curandState
    curand_init(43, id, 0, &rngState[id]);
}

__host__
extern "C" void cudaInitializePyramidRNGStates(curandState** rngStateOnGPU, const int numCandidatesToFind) {
    int numBlocks = numCandidatesToFind / THREADS_PER_BLOCK + (numCandidatesToFind % THREADS_PER_BLOCK == 0 ? 0 : 1);
    int numThreads = numBlocks * THREADS_PER_BLOCK;
    cudaCheckSuccess(cudaMalloc(rngStateOnGPU, sizeof(curandState) * numThreads));
    cuda_init_curand_statePyramid <<< numBlocks, THREADS_PER_BLOCK >>> (*rngStateOnGPU);
    cudaDeviceSynchronize();
    cudaCheckExecution();
}

__host__
extern "C" void cudaLaunchImportanceSamplingKernel(
    uint3* candidates,
    const int numCandidatesToFind,
    const REAL* importancePyramid,
    const LevelStats* levelStats,
    curandState* rngStateOnGPU,
    const int topLevel
) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    int updatePhaseBatchSize = deviceProperties.multiProcessorCount * 4;
    int numBlocks = numCandidatesToFind / THREADS_PER_BLOCK + (numCandidatesToFind % THREADS_PER_BLOCK == 0 ? 0 : 1);

    // setup curand
    cuda_selectImportanceSamplingCandidates << < numBlocks, THREADS_PER_BLOCK >> > (candidates, importancePyramid, levelStats, rngStateOnGPU, topLevel, BLOCK_SIZE);
    cudaDeviceSynchronize();
    cudaCheckExecution();
}


