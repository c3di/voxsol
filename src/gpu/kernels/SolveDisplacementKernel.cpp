#pragma once
#include "stdafx.h"
#include "SolveDisplacementKernel.h"

SolveDisplacementKernel::SolveDisplacementKernel(Solution* sol) :
    solution(sol),
    serializedMatConfigEquations(nullptr),
    serializedVertices(nullptr)
{

};

SolveDisplacementKernel::~SolveDisplacementKernel() {
    freeCudaResources();
    assert(serializedMatConfigEquations == nullptr);
    assert(serializedVertices == nullptr);
};


void SolveDisplacementKernel::launch() {
    prepareInputs();

    if (canExecute()) {
        unsigned int numVertices = static_cast<unsigned int>(solution->getVertices()->size());
        cudaLaunchSolveDisplacementKernel(serializedVertices, serializedMatConfigEquations, numVertices);
        cudaDeviceSynchronize();
        pullVertices();
    }
};

bool SolveDisplacementKernel::canExecute() {
    if (serializedMatConfigEquations == nullptr || serializedVertices == nullptr) {
        throw std::runtime_error("Could not execute kernel SolveDisplacement because one or more inputs are missing.");
    }

    return true;
};

void SolveDisplacementKernel::freeCudaResources() {
    cudaCheckSuccess(cudaFree(serializedMatConfigEquations));
    serializedMatConfigEquations = nullptr;
    cudaCheckSuccess(cudaFree(serializedVertices));
    serializedVertices = nullptr;
}

void SolveDisplacementKernel::prepareInputs() {
    pushMatConfigEquations();
    pushVertices();
}

void SolveDisplacementKernel::pushMatConfigEquations() {
    size_t size = solution->getMaterialConfigurationEquations()->size() * MaterialConfigurationEquations::SizeInBytes;
    cudaCheckSuccess(cudaMallocManaged(&serializedMatConfigEquations, size));
    serializeMaterialConfigurationEquations(serializedMatConfigEquations);
};

void SolveDisplacementKernel::pushVertices() {
    const std::vector<Vertex>* vertices = solution->getVertices();
    size_t size = vertices->size() * sizeof(Vertex);
    cudaCheckSuccess(cudaMallocManaged(&serializedVertices, size));
    memcpy(serializedVertices, vertices->data(), size);
}

void SolveDisplacementKernel::pullVertices() {
    std::vector<Vertex>* vertices = solution->getVertices();
    size_t size = vertices->size() * sizeof(Vertex);
    memcpy(vertices->data(), serializedVertices, size);
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
