#pragma once
#include "stdafx.h"
#include "SolveDisplacementKernel.h"

SolveDisplacementKernel::SolveDisplacementKernel(Solution* sol) :
    solution(sol),
    displacementsOnGPU(nullptr),
    matConfigEquationIdsOnGPU(nullptr),
    matConfigEquationsOnGPU(nullptr)
{

};

SolveDisplacementKernel::~SolveDisplacementKernel() {
    freeCudaResources();
    assert(matConfigEquationIdsOnGPU == nullptr);
    assert(displacementsOnGPU == nullptr);
    assert(matConfigEquationsOnGPU == nullptr);
};


void SolveDisplacementKernel::launch() {
    prepareInputs();

    if (canExecute()) {
        unsigned int numVertices = static_cast<unsigned int>(solution->getDisplacements()->size());
        cudaLaunchSolveDisplacementKernel(displacementsOnGPU, matConfigEquationIdsOnGPU, matConfigEquationsOnGPU, numVertices);

        pullDisplacements();
    }
};

bool SolveDisplacementKernel::canExecute() {
    if (matConfigEquationIdsOnGPU == nullptr || displacementsOnGPU == nullptr || matConfigEquationsOnGPU == nullptr) {
        throw std::runtime_error("Could not execute kernel SolveDisplacement because one or more inputs are missing.");
    }

    return true;
};

void SolveDisplacementKernel::freeCudaResources() {
    cudaCheckSuccess(cudaFree(matConfigEquationIdsOnGPU));
    matConfigEquationIdsOnGPU = nullptr;
    cudaCheckSuccess(cudaFree(displacementsOnGPU));
    displacementsOnGPU = nullptr;
    cudaCheckSuccess(cudaFree(matConfigEquationsOnGPU));
    matConfigEquationsOnGPU = nullptr;
}

void SolveDisplacementKernel::prepareInputs() {
    pushMatConfigEquationIds();
    pushDisplacements();
    pushMatConfigEquations();
}

void SolveDisplacementKernel::pushMatConfigEquationIds() {
    const std::vector<ConfigId>* signatureIds = solution->getMaterialConfigurationEquationIds();
    size_t size = signatureIds->size() * sizeof(ConfigId);
    cudaCheckSuccess(cudaMalloc(&matConfigEquationIdsOnGPU, size));
    cudaCheckSuccess(cudaMemcpy(matConfigEquationIdsOnGPU, signatureIds->data(), size, cudaMemcpyHostToDevice));
};

void SolveDisplacementKernel::pushDisplacements() {
    const std::vector<REAL>* displacements = solution->getDisplacements();
    size_t size = displacements->size() * sizeof(REAL);
    cudaCheckSuccess(cudaMalloc(&displacementsOnGPU, size));
    cudaCheckSuccess(cudaMemcpy(displacementsOnGPU, displacements->data(), size, cudaMemcpyHostToDevice));
};

void SolveDisplacementKernel::pushMatConfigEquations() {
    size_t size = solution->getMaterialConfigurationEquations()->size() * MaterialConfigurationEquations::SizeInBytes;
    void* h_matConfigEquations = malloc(size);
    serializeMaterialConfigurationEquations(h_matConfigEquations);

    cudaCheckSuccess(cudaMalloc(&matConfigEquationsOnGPU, size));
    cudaCheckSuccess(cudaMemcpy(matConfigEquationsOnGPU, h_matConfigEquations, size, cudaMemcpyHostToDevice));

    delete[] h_matConfigEquations;
};

void SolveDisplacementKernel::pullDisplacements() {
    std::vector<REAL>* displacements = solution->getDisplacements();
    size_t size = displacements->size() * sizeof(REAL);
    cudaCheckSuccess(cudaMemcpy(displacements->data(), displacementsOnGPU, size, cudaMemcpyDeviceToHost));
};

void SolveDisplacementKernel::serializeMaterialConfigurationEquations(void* destination) {
    const std::vector<MaterialConfigurationEquations>* signatures = solution->getMaterialConfigurationEquations();
    size_t size = MaterialConfigurationEquations::SizeInBytes * signatures->size();

    char* serializationPointer = (char*)destination;
    for (unsigned int i = 0; i < signatures->size(); i++) {
        signatures->at(i).serialize(serializationPointer);
        serializationPointer += MaterialConfigurationEquations::SizeInBytes;
    }
}
