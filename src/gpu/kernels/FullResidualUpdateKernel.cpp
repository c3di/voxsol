#include "stdafx.h"
#include "FullResidualUpdateKernel.h"
#include "gpu/GPUParameters.h"

FullResidualUpdateKernel::FullResidualUpdateKernel(Solution* sol, ResidualVolume* residualVolume, Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU) :
    solution(sol),
    residualVolume(residualVolume),
    verticesOnGPU(verticesOnGPU),
    matConfigEquationsOnGPU(matConfigEquationsOnGPU)
{
    solutionDimensions.x = sol->getSize().x;
    solutionDimensions.y = sol->getSize().y;
    solutionDimensions.z = sol->getSize().z;

    residualLevelZeroDimensions.x = residualVolume->getPointerToStatsForLevel(0)->sizeX;
    residualLevelZeroDimensions.y = residualVolume->getPointerToStatsForLevel(0)->sizeY;
    residualLevelZeroDimensions.z = residualVolume->getPointerToStatsForLevel(0)->sizeZ;
}

FullResidualUpdateKernel::~FullResidualUpdateKernel() {

}

void FullResidualUpdateKernel::setMatConfigEquationsOnGPU(REAL* equationsOnGPU) {
    matConfigEquationsOnGPU = equationsOnGPU;
}

void FullResidualUpdateKernel::setVerticesOnGPU(Vertex* vertices) {
    verticesOnGPU = vertices;
}

void FullResidualUpdateKernel::launch() {
    if (!canExecute()) {
        throw std::exception("Could not execute full residual update kernel");
    }

    cudaLaunchFullResidualUpdateKernel(verticesOnGPU, residualVolume->getPyramidDevicePointer(), matConfigEquationsOnGPU, solutionDimensions);

}

bool FullResidualUpdateKernel::canExecute() {
    return matConfigEquationsOnGPU != nullptr && verticesOnGPU != nullptr;
}

void FullResidualUpdateKernel::freeCudaResources() {

}
