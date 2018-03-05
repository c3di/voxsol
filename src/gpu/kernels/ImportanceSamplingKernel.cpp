#pragma once
#include "stdafx.h"
#include "ImportanceSamplingKernel.h"
#include <iostream>
#include <fstream>

ImportanceSamplingKernel::ImportanceSamplingKernel(ResidualVolume* resVol) :
    residualVolume(resVol),
    numBlocksToGenerate(0),
    blockOrigins(nullptr),
    rngStateOnGPU(nullptr)
{
}

ImportanceSamplingKernel::~ImportanceSamplingKernel()
{
    freeCudaResources();
}

void ImportanceSamplingKernel::launch() {
    if (!canExecute()) {
        initCurandState();
    }
    if (canExecute()) {
        cudaLaunchPyramidUpdateKernel(residualVolume->getPyramidDevicePointer(), residualVolume->getNumberOfLevels(), residualVolume->getLevelStatsDevicePointer());

        cudaLaunchImportanceSamplingKernel(
            blockOrigins, 
            numBlocksToGenerate, 
            residualVolume->getPyramidDevicePointer(),
            residualVolume->getLevelStatsDevicePointer(),
            rngStateOnGPU,
            residualVolume->getNumberOfLevels() - 1
        );
        cudaDeviceSynchronize();
    }
    else {
        throw "Importance sampling kernel could not be executed";
    }
}

void ImportanceSamplingKernel::setBlockOriginsDestination(uint3* dest) {
    blockOrigins = dest;
}

void ImportanceSamplingKernel::setNumBlocksToFind(int numBlocks) {
    if (numBlocks != numBlocksToGenerate) {
        numBlocksToGenerate = numBlocks;
        initCurandState();
    }
}

bool ImportanceSamplingKernel::canExecute() {
    return blockOrigins != nullptr && rngStateOnGPU != nullptr;
}

void ImportanceSamplingKernel::freeCudaResources()
{
    // Note: blockOrigins pointer is provided externally, should not be cleaned up here
    if (rngStateOnGPU != nullptr) {
        cudaCheckSuccess(cudaFree(rngStateOnGPU));
        rngStateOnGPU = nullptr;
    }
}

void ImportanceSamplingKernel::initCurandState() {
    cudaInitializePyramidRNGStates(&rngStateOnGPU, numBlocksToGenerate);
}
