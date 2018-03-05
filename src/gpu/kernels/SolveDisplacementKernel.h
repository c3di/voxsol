#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>
#include <curand_kernel.h>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "gpu/sampling/ResidualVolume.h"
#include "solution/samplers/BlockSampler.h"

//extern "C" void cudaLaunchSolveDisplacementKernelGlobal(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, Vertex* blockOrigins, const SolutionDim solutionDims);
extern "C" void cudaLaunchSolveDisplacementKernelGlobalResiduals(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, REAL* residualVolume, curandState* rngStateOnGPU, uint3* blockOrigins, const int numBlocks, const uint3 solutionDims, const LevelStats* levelStats);
//extern "C" void cudaLaunchSolveDisplacementKernelShared(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, REAL* importanceVolume, curandState* rngStateOnGPU, uint3* blockOrigins, const int numBlocks, const uint3 solutionDims, const LevelStats* levelStats);
extern "C" void cudaInitializeRNGStatesGlobal(curandState** rngState);

class SolveDisplacementKernel : public CudaKernel {

public:

    SolveDisplacementKernel(Solution* sol, BlockSampler* sampler, ResidualVolume* resVol);
    ~SolveDisplacementKernel();

    void launch() override;
    void solveCPU();

    void pullVertices();

    //This returns a pointer to managed memory, use only for debug output to avoid unnecessary cpu/gpu memory paging!
    uint3* debugGetImportanceSamplesManaged();
    void debugOutputEquationsCPU();
    void debugOutputEquationsGPU();

protected:

    bool canExecute() override;
    void freeCudaResources();

    Solution* solution;
    uint3 solutionDimensions;
    ResidualVolume* residualVolume;
    BlockSampler* sampler;

    Vertex* serializedVertices;
    REAL* serializedMatConfigEquations;
    uint3* blockOrigins;
    curandState* rngStateOnGPU;

    void prepareInputs();

    void pushMatConfigEquationsManaged();
    void pushVerticesManaged();
    void initCurandState();
    void allocateBlockOrigins();

    void serializeMaterialConfigurationEquations(void* destination);

    void cpuBuildRHSVector(libmmv::Vec3<REAL>* rhsVec, const MaterialConfigurationEquations* matrices, int x, int y, int z);
    void cpuSolveIteration();
};
