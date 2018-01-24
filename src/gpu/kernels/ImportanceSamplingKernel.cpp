#pragma once
#include "stdafx.h"
#include "ImportanceSamplingKernel.h"
#include <iostream>
#include <fstream>

ImportanceSamplingKernel::ImportanceSamplingKernel(ImportanceVolume * impVol, unsigned int numCandidates) : 
    importanceVolume(impVol),
    numberOfCandidatesToFind(numCandidates),
    blockOrigins(nullptr)
{
}

ImportanceSamplingKernel::~ImportanceSamplingKernel()
{
    freeCudaResources();
}

void ImportanceSamplingKernel::launch()
{
    if (!canExecute()) {
        allocateCandidatesArray();
    }
    if (canExecute()) {
        cudaLaunchImportanceSamplingKernel(
            blockOrigins, 
            numberOfCandidatesToFind, 
            importanceVolume->getPyramidDevicePointer(), 
            importanceVolume->getLevelStatsDevicePointer(), 
            importanceVolume->getNumberOfLevels() - 1
        );
        cudaDeviceSynchronize();
    }
}

bool ImportanceSamplingKernel::canExecute()
{
    return blockOrigins != nullptr;
}

uint3* ImportanceSamplingKernel::getBlockOriginsDevicePointer() {
    return blockOrigins;
}

void ImportanceSamplingKernel::freeCudaResources()
{
    if (blockOrigins != nullptr) {
        cudaCheckSuccess(cudaFree(blockOrigins));
        blockOrigins = nullptr;
    }
}

void ImportanceSamplingKernel::allocateCandidatesArray() {
    size_t size = numberOfCandidatesToFind * sizeof(Vertex);
    cudaCheckSuccess(cudaMallocManaged(&blockOrigins, size));
}

void ImportanceSamplingKernel::setNumberOfCandidatesToFind(unsigned int numCandidates) {
    numberOfCandidatesToFind = numCandidates;
    freeCudaResources();
    allocateCandidatesArray();
}
