#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>
#include <curand_kernel.h>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "gpu/sampling/ImportanceVolume.h"
#include "gpu/kernels/ImportanceSamplingKernel.h"

//extern "C" void cudaLaunchSolveDisplacementKernelGlobal(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, Vertex* blockOrigins, const SolutionDim solutionDims);
extern "C" void cudaLaunchSolveDisplacementKernelGlobalResiduals(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, REAL* importanceVolume, curandState* rngStateOnGPU, uint3* blockOrigins, const int numBlocks, const uint3 solutionDims, const LevelStats* levelStats);
extern "C" void cudaLaunchPyramidUpdateKernel(REAL* importancePyramid, const int numLevels, const LevelStats* levelStats);
extern "C" void cudaInitializeRNGStatesGlobal(curandState** rngState);

class SolveDisplacementKernel : public CudaKernel {

public:

    SolveDisplacementKernel(Solution* sol, ImportanceVolume* impVol);
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
    ImportanceSamplingKernel importanceSampler;
    ImportanceVolume* importanceVolume;

    Vertex* serializedVertices;
    REAL* serializedMatConfigEquations;
    curandState* rngStateOnGPU;

    void prepareInputs();

    void pushMatConfigEquationsManaged();
    void pushVerticesManaged();
    void initCurandState();

    void serializeMaterialConfigurationEquations(void* destination);

    void cpuBuildRHSVector(libmmv::Vec3<REAL>* rhsVec, const MaterialConfigurationEquations* matrices, int x, int y, int z);
    void cpuSolveIteration();
};
