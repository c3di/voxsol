#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "problem/Material.h"

#define cudaCheckSuccess(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

class CK_ComputeLHSMatrices : public CudaKernel {

    struct MaterialStruct {
        unsigned int id;
        REAL mu;
        REAL lambda;
    };


public:

    CK_ComputeLHSMatrices(ettention::Vec3ui& voxelDims) :
        m_voxelDimensions(voxelDims)
    {

    };

    ~CK_ComputeLHSMatrices() {
        assert(di_linearBaseIntegrals == nullptr);
        assert(di_quadBaseIntegrals == nullptr);
        assert(di_materialIds == nullptr);
        assert(di_materials == nullptr);
        assert(do_lhsMatrices == nullptr);
    };

    void setInput_LinearIntegrals(REAL* data, size_t size) {
        hi_linearBaseIntegrals = data;
        cudaCheckSuccess(cudaMalloc(&di_linearBaseIntegrals, size));
        cudaCheckSuccess(cudaMemcpy(di_linearBaseIntegrals, data, size, cudaMemcpyHostToDevice));
    };

    void setInput_QuadIntegrals(REAL* data, size_t size) {
        hi_quadBaseIntegrals = data;
        cudaCheckSuccess(cudaMalloc(&di_quadBaseIntegrals, size));
        cudaCheckSuccess(cudaMemcpy(di_quadBaseIntegrals, data, size, cudaMemcpyHostToDevice));
    };

    void setInput_MaterialIds(unsigned int* data, size_t size) {
        if (size != sizeof(unsigned int) * m_voxelDimensions.x * m_voxelDimensions.y * m_voxelDimensions.z) {
            throw std::invalid_argument("size of material id data does not match expected size");
        }
        hi_materialIds = data;
        cudaCheckSuccess(cudaMalloc(&di_materialIds, size));
        cudaCheckSuccess(cudaMemcpy(di_materialIds, data, size, cudaMemcpyHostToDevice));
    };

    void setInput_Materials(std::vector<Material>* data) {
        for (auto it = data->begin(); it != data->end(); ++it) {
            MaterialStruct mat;
            mat.id = it->m_id;
            mat.lambda = it->m_lambda;
            mat.mu = it->m_mu;
            hi_materials->push_back(mat);
        }

        size_t size = sizeof(MaterialStruct) * hi_materials->size();

        cudaCheckSuccess(cudaMalloc(&di_materials, size));
        cudaCheckSuccess(cudaMemcpy(di_materials, hi_materials->data(), sizeof(MaterialStruct) * hi_materials->size(), cudaMemcpyHostToDevice));
    };

    void setOutput_LHSMatrices(REAL* data, size_t expectedSize) {
        ho_lhsMatrices = data;
        cudaCheckSuccess(cudaMalloc(&ho_lhsMatrices, expectedSize));
    };

    void execute() override {
        if (canExecute()) {
            
        }
    };

protected:

    bool canExecute() override {
        assert(di_linearBaseIntegrals != nullptr);
        assert(di_quadBaseIntegrals != nullptr);
        assert(di_materialIds != nullptr);
        assert(di_materials != nullptr);
        assert(do_lhsMatrices != nullptr);

        return true;
    };

    void freeCudaResources() {
        cudaCheckSuccess(cudaFree(di_linearBaseIntegrals));
        di_linearBaseIntegrals = nullptr;
        cudaCheckSuccess(cudaFree(di_quadBaseIntegrals));
        di_quadBaseIntegrals = nullptr;
        cudaCheckSuccess(cudaFree(di_materialIds));
        di_materialIds = nullptr;
        cudaCheckSuccess(cudaFree(di_materials));
        di_materials = nullptr;
        cudaCheckSuccess(cudaFree(do_lhsMatrices));
        do_lhsMatrices = nullptr;
    }

private:
    const ettention::Vec3ui m_voxelDimensions;

    REAL* hi_linearBaseIntegrals;
    REAL* hi_quadBaseIntegrals;
    unsigned int* hi_materialIds;
    std::vector<MaterialStruct>* hi_materials;
    REAL* ho_lhsMatrices;

    REAL* di_linearBaseIntegrals;
    REAL* di_quadBaseIntegrals;
    unsigned int* di_materialIds;
    std::vector<MaterialStruct>* di_materials;
    REAL* do_lhsMatrices;

};
