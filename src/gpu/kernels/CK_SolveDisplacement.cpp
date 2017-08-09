#pragma once
#include "stdafx.h"
#include "CK_SolveDisplacement.h"

CK_SolveDisplacement::CK_SolveDisplacement(Solution* sol) :
    solution(sol),
    d_displacements(nullptr),
    d_matConfigEquationIds(nullptr),
    d_matConfigEquations(nullptr)
{

};

CK_SolveDisplacement::~CK_SolveDisplacement() {
    freeCudaResources();
    assert(d_matConfigEquationIds == nullptr);
    assert(d_displacements == nullptr);
    assert(d_matConfigEquations == nullptr);
};


void CK_SolveDisplacement::launchKernel() {
    prepareInputs();

    if (canExecute()) {
        unsigned int numVertices = static_cast<unsigned int>(solution->getDisplacements()->size());
        CK_SolveDisplacement_launch(d_displacements, d_matConfigEquationIds, d_matConfigEquations, numVertices);

        pull_displacements();
    }
};

bool CK_SolveDisplacement::canExecute() {
    if (d_matConfigEquationIds == nullptr || d_displacements == nullptr || d_matConfigEquations == nullptr) {
        throw std::runtime_error("Could not execute kernel SolveDisplacement because one or more inputs are missing.");
    }

    return true;
};

void CK_SolveDisplacement::freeCudaResources() {
    cudaCheckSuccess(cudaFree(d_matConfigEquationIds));
    d_matConfigEquationIds = nullptr;
    cudaCheckSuccess(cudaFree(d_displacements));
    d_displacements = nullptr;
    cudaCheckSuccess(cudaFree(d_matConfigEquations));
    d_matConfigEquations = nullptr;
}

void CK_SolveDisplacement::prepareInputs() {
    push_matConfigEquationIds();
    push_displacements();
    push_matConfigEquations();
}

void CK_SolveDisplacement::push_matConfigEquationIds() {
    const std::vector<unsigned short>* signatureIds = solution->getMaterialConfigurationEquationIds();
    size_t size = signatureIds->size() * sizeof(unsigned short);
    cudaCheckSuccess(cudaMalloc(&d_matConfigEquationIds, size));
    cudaCheckSuccess(cudaMemcpy(d_matConfigEquationIds, signatureIds->data(), size, cudaMemcpyHostToDevice));
};

void CK_SolveDisplacement::push_displacements() {
    const std::vector<REAL>* displacements = solution->getDisplacements();
    size_t size = displacements->size() * sizeof(REAL);
    cudaCheckSuccess(cudaMalloc(&d_displacements, size));
    cudaCheckSuccess(cudaMemcpy(d_displacements, displacements->data(), size, cudaMemcpyHostToDevice));
};

void CK_SolveDisplacement::push_matConfigEquations() {
    size_t size = solution->getMaterialConfigurationEquations()->size() * MaterialConfigurationEquations::SizeInBytes;
    void* h_matConfigEquations = malloc(size);
    serializeMaterialConfigurationEquations(h_matConfigEquations);

    cudaCheckSuccess(cudaMalloc(&d_matConfigEquations, size));
    cudaCheckSuccess(cudaMemcpy(d_matConfigEquations, h_matConfigEquations, size, cudaMemcpyHostToDevice));

    delete[] h_matConfigEquations;
};

void CK_SolveDisplacement::pull_displacements() {
    std::vector<REAL>* displacements = solution->getDisplacements();
    size_t size = displacements->size() * sizeof(REAL);
    cudaCheckSuccess(cudaMemcpy(displacements->data(), d_displacements, size, cudaMemcpyDeviceToHost));
};

void CK_SolveDisplacement::serializeMaterialConfigurationEquations(void* destination) {
    const std::vector<MaterialConfigurationEquations>* signatures = solution->getMaterialConfigurationEquations();
    size_t size = MaterialConfigurationEquations::SizeInBytes * signatures->size();

    char* serializationPointer = (char*)destination;
    for (unsigned int i = 0; i < signatures->size(); i++) {
        signatures->at(i).serialize(serializationPointer);
        serializationPointer += MaterialConfigurationEquations::SizeInBytes;
    }
}
