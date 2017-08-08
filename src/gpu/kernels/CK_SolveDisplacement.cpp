#pragma once
#include "stdafx.h"
#include "CK_SolveDisplacement.h"

CK_SolveDisplacement::CK_SolveDisplacement(Solution* sol) :
    m_solution(sol),
    d_displacements(nullptr),
    d_signatureIds(nullptr),
    d_fragmentSignatures(nullptr)
{

};

CK_SolveDisplacement::~CK_SolveDisplacement() {
    freeCudaResources();
    assert(d_signatureIds == nullptr);
    assert(d_displacements == nullptr);
    assert(d_fragmentSignatures == nullptr);
};


void CK_SolveDisplacement::launchKernel() {
    prepareInputs();

    if (canExecute()) {
        unsigned int numVertices = static_cast<unsigned int>(m_solution->getDisplacements()->size());
        CK_SolveDisplacement_launch(d_displacements, d_signatureIds, d_fragmentSignatures, numVertices);

        pull_displacements();
    }
};

bool CK_SolveDisplacement::canExecute() {
    assert(d_signatureIds != nullptr);
    assert(d_displacements != nullptr);
    assert(d_fragmentSignatures != nullptr);

    return true;
};

void CK_SolveDisplacement::freeCudaResources() {
    cudaCheckSuccess(cudaFree(d_signatureIds));
    d_signatureIds = nullptr;
    cudaCheckSuccess(cudaFree(d_displacements));
    d_displacements = nullptr;
    cudaCheckSuccess(cudaFree(d_fragmentSignatures));
    d_fragmentSignatures = nullptr;
}

void CK_SolveDisplacement::prepareInputs() {
    push_signatureIds();
    push_displacements();
    push_fragmentSignatures();
}

void CK_SolveDisplacement::push_signatureIds() {
    const std::vector<unsigned short>* signatureIds = m_solution->getSignatureIds();
    size_t size = signatureIds->size() * sizeof(unsigned short);
    cudaCheckSuccess(cudaMalloc(&d_signatureIds, size));
    cudaCheckSuccess(cudaMemcpy(d_signatureIds, signatureIds->data(), size, cudaMemcpyHostToDevice));
};

void CK_SolveDisplacement::push_displacements() {
    const std::vector<REAL>* displacements = m_solution->getDisplacements();
    size_t size = displacements->size() * sizeof(REAL);
    cudaCheckSuccess(cudaMalloc(&d_displacements, size));
    cudaCheckSuccess(cudaMemcpy(d_displacements, displacements->data(), size, cudaMemcpyHostToDevice));
};

void CK_SolveDisplacement::push_fragmentSignatures() {
    size_t size = m_solution->getFragmentSignatures()->size() * FragmentSignature::SizeInBytes;
    void* h_fragmentSignatures = malloc(size);
    serializeFragmentSignatures(h_fragmentSignatures);

    cudaCheckSuccess(cudaMalloc(&d_fragmentSignatures, size));
    cudaCheckSuccess(cudaMemcpy(d_fragmentSignatures, h_fragmentSignatures, size, cudaMemcpyHostToDevice));

    delete[] h_fragmentSignatures;
};

void CK_SolveDisplacement::pull_displacements() {
    std::vector<REAL>* displacements = m_solution->getDisplacements();
    size_t size = displacements->size() * sizeof(REAL);
    cudaCheckSuccess(cudaMemcpy(displacements->data(), d_displacements, size, cudaMemcpyDeviceToHost));
};

void CK_SolveDisplacement::serializeFragmentSignatures(void* destination) {
    const std::vector<FragmentSignature>* signatures = m_solution->getFragmentSignatures();
    size_t size = FragmentSignature::SizeInBytes * signatures->size();

    char* serializationPointer = (char*)destination;
    for (unsigned int i = 0; i < signatures->size(); i++) {
        signatures->at(i).serialize(serializationPointer);
        serializationPointer += FragmentSignature::SizeInBytes;
    }
}
