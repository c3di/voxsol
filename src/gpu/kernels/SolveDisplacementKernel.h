#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "gpu/sampling/ImportanceVolume.h"
#include "gpu/kernels/ImportanceSamplingKernel.h"

//extern "C" void cudaLaunchSolveDisplacementKernelGlobal(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, Vertex* blockOrigins, const SolutionDim solutionDims);
extern "C" void cudaLaunchSolveDisplacementKernelGlobalResiduals(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, REAL* importanceVolume, uint3* blockOrigins, const int numBlocks, const uint3 solutionDims);
extern "C" void cudaLaunchPyramidUpdateKernel(REAL* importancePyramid, const int numLevels, const LevelStats* levelStats);

class SolveDisplacementKernel : public CudaKernel {

public:

    SolveDisplacementKernel(Solution* sol, ImportanceVolume* impVol);
    ~SolveDisplacementKernel();

    void launch() override;
    void solveCPU();

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

    void prepareInputs();

    void pushMatConfigEquationsManaged();
    void pushVerticesManaged();

    void pullVertices();

    void serializeMaterialConfigurationEquations(void* destination);

    void cpuBuildRHSVector(libmmv::Vec3<REAL>* rhsVec, const MaterialConfigurationEquations* matrices, int x, int y, int z);
    void cpuSolveIteration();
};
