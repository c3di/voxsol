#pragma once
#include "stdafx.h"
#include "SolveDisplacementKernel.h"

SolveDisplacementKernel::SolveDisplacementKernel(Solution* sol) :
    solution(sol),
    matConfigEquationsOnGPU(nullptr),
    verticesOnGPU(nullptr)
{

};

SolveDisplacementKernel::~SolveDisplacementKernel() {
    freeCudaResources();
    assert(matConfigEquationsOnGPU == nullptr);
    assert(verticesOnGPU == nullptr);
};


void SolveDisplacementKernel::launch() {
    prepareInputs();

    if (canExecute()) {
        unsigned int numVertices = static_cast<unsigned int>(solution->getVertices()->size());
        cudaLaunchSolveDisplacementKernel(verticesOnGPU, matConfigEquationsOnGPU, numVertices);

        pullVertices();
    }
};

bool SolveDisplacementKernel::canExecute() {
    if (matConfigEquationsOnGPU == nullptr || verticesOnGPU == nullptr) {
        throw std::runtime_error("Could not execute kernel SolveDisplacement because one or more inputs are missing.");
    }

    return true;
};

void SolveDisplacementKernel::freeCudaResources() {
    cudaCheckSuccess(cudaFree(matConfigEquationsOnGPU));
    matConfigEquationsOnGPU = nullptr;
    cudaCheckSuccess(cudaFree(verticesOnGPU));
    verticesOnGPU = nullptr;
}

void SolveDisplacementKernel::prepareInputs() {
    pushMatConfigEquations();
    pushVertices();
}

void SolveDisplacementKernel::pushMatConfigEquations() {
    size_t size = solution->getMaterialConfigurationEquations()->size() * MaterialConfigurationEquations::SizeInBytes;
    void* h_matConfigEquations = malloc(size);
    serializeMaterialConfigurationEquations(h_matConfigEquations);

    cudaCheckSuccess(cudaMalloc(&matConfigEquationsOnGPU, size));
    cudaCheckSuccess(cudaMemcpy(matConfigEquationsOnGPU, h_matConfigEquations, size, cudaMemcpyHostToDevice));

    delete[] h_matConfigEquations;
};

void SolveDisplacementKernel::pushVertices() {
    const std::vector<Vertex>* vertices = solution->getVertices();
    size_t size = vertices->size() * sizeof(Vertex);
    cudaCheckSuccess(cudaMalloc(&verticesOnGPU, size));
    cudaCheckSuccess(cudaMemcpy(verticesOnGPU, vertices->data(), size, cudaMemcpyHostToDevice));
}

void SolveDisplacementKernel::pullVertices() {
    std::vector<Vertex>* vertices = solution->getVertices();
    size_t size = vertices->size() * sizeof(Vertex);
    cudaCheckSuccess(cudaMemcpy(vertices->data(), verticesOnGPU, size, cudaMemcpyDeviceToHost));
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
