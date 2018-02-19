#pragma once
#include "stdafx.h"
#include "ImportanceSamplingKernel.h"
#include <iostream>
#include <fstream>

ImportanceSamplingKernel::ImportanceSamplingKernel(ImportanceVolume * impVol, unsigned int numCandidates) :
    importanceVolume(impVol),
    numberOfCandidatesToFind(numCandidates),
    blockOrigins(nullptr),
    rngStateOnGPU(nullptr)
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
        initCurandState();
    }
    if (canExecute()) {
        cudaLaunchImportanceSamplingKernel(
            blockOrigins, 
            numberOfCandidatesToFind, 
            importanceVolume->getPyramidDevicePointer(), 
            importanceVolume->getLevelStatsDevicePointer(), 
            rngStateOnGPU,
            importanceVolume->getNumberOfLevels() - 1
        );
        cudaDeviceSynchronize();
    }
}

bool ImportanceSamplingKernel::canExecute()
{
    return blockOrigins != nullptr && rngStateOnGPU != nullptr;
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
    if (rngStateOnGPU != nullptr) {
        cudaCheckSuccess(cudaFree(rngStateOnGPU));
        rngStateOnGPU = nullptr;
    }
}

void ImportanceSamplingKernel::initCurandState() {
    cudaInitializePyramidRNGStates(&rngStateOnGPU, numberOfCandidatesToFind);
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
