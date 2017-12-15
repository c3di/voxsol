#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "solution/samplers/RandomBlockSampler.h"

extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, BlockSampler& sampler, const SolutionDim solutionDims);

class SolveDisplacementKernel : public CudaKernel {

public:

    SolveDisplacementKernel(Solution* sol);
    ~SolveDisplacementKernel();

    void launch() override;
    void solveCPU();

    void debugOutputEquationsCPU();
    void debugOutputEquationsGPU();

protected:

    bool canExecute() override;
    void freeCudaResources();

private:
    Solution* solution;
    SolutionDim solutionDimensions;
    RandomBlockSampler sampler;

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
