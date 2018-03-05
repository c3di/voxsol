#include "stdafx.h"
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "gpu/sampling/ResidualVolume.h"

__device__
void addResidualFromLowerLevelVertex(
    unsigned int x,
    unsigned int y,
    unsigned int z,
    REAL* residual,
    REAL* importancePyramid,
    const LevelStats& lowerLevelStats
) {
    if (z >= lowerLevelStats.sizeZ || y >= lowerLevelStats.sizeY || x >= lowerLevelStats.sizeX) {
        return;
    }
    int coordIndex = z * lowerLevelStats.sizeX * lowerLevelStats.sizeY + y * lowerLevelStats.sizeX + x;
    REAL* lowerLevelResidual = importancePyramid + lowerLevelStats.startIndex + coordIndex;
    *residual += *lowerLevelResidual;
}

__global__
void cuda_updatePyramidLevel(REAL* importancePyramid, const int activeLevel, const LevelStats* levelStats) {
    int residualX = blockIdx.x * blockDim.x + threadIdx.x;
    int residualY = blockIdx.y * blockDim.y + threadIdx.y;
    int residualZ = blockIdx.z * blockDim.z + threadIdx.z;
    LevelStats activeLevelStats = levelStats[activeLevel];
    LevelStats lowerLevelStats = levelStats[activeLevel - 1];

    if (residualX >= activeLevelStats.sizeX || residualY >= activeLevelStats.sizeY || residualZ >= activeLevelStats.sizeZ) {
        return;
    }

    int residualXProjected = residualX * 2;
    int residualYProjected = residualY * 2;
    int residualZProjected = residualZ * 2;  
    REAL residual = asREAL(0.0);

    addResidualFromLowerLevelVertex(residualXProjected, residualYProjected, residualZProjected, &residual, importancePyramid, lowerLevelStats);
    addResidualFromLowerLevelVertex(residualXProjected + 1, residualYProjected, residualZProjected, &residual, importancePyramid, lowerLevelStats);
    addResidualFromLowerLevelVertex(residualXProjected, residualYProjected + 1, residualZProjected, &residual, importancePyramid, lowerLevelStats);
    addResidualFromLowerLevelVertex(residualXProjected + 1, residualYProjected + 1, residualZProjected, &residual, importancePyramid, lowerLevelStats);

    addResidualFromLowerLevelVertex(residualXProjected, residualYProjected, residualZProjected + 1, &residual, importancePyramid, lowerLevelStats);
    addResidualFromLowerLevelVertex(residualXProjected + 1, residualYProjected, residualZProjected + 1, &residual, importancePyramid, lowerLevelStats);
    addResidualFromLowerLevelVertex(residualXProjected, residualYProjected + 1, residualZProjected + 1, &residual, importancePyramid, lowerLevelStats);
    addResidualFromLowerLevelVertex(residualXProjected + 1, residualYProjected + 1, residualZProjected + 1, &residual, importancePyramid, lowerLevelStats);

    REAL* residualToUpdate = importancePyramid + activeLevelStats.startIndex;
    residualToUpdate += residualZ * activeLevelStats.sizeX * activeLevelStats.sizeY + residualY * activeLevelStats.sizeX + residualX;
    *residualToUpdate = residual;
}

// Note: activeLevel is always >=1 because level 0 is updated by the solve displacement kernel
__host__
extern "C" void cudaLaunchPyramidUpdateKernel(REAL* importancePyramid, const int numLevels, const LevelStats* levelStats) {
    // setup execution parameters
    // TODO: hold this data on both CPU and GPU separately to avoid paging
    for (int i = 1; i < numLevels; i++) {
        LevelStats activeLevelStats = levelStats[i];
        dim3 threadsPerBlock(8, 8, 8);
        dim3 grid(activeLevelStats.sizeX / 8 + 1, activeLevelStats.sizeY / 8 + 1, activeLevelStats.sizeZ / 8 + 1);

        cuda_updatePyramidLevel << < grid, threadsPerBlock >> >(importancePyramid, i, levelStats);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }    
}


