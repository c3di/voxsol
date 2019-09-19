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
#include "FullResidualUpdateKernel.h"
#include "gpu/GPUParameters.h"

extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, int3* blockOrigins, const int numBlocks, const uint3 solutionDims);

class SolveDisplacementKernel : public CudaKernel {

public:

    SolveDisplacementKernel(Solution* sol, BlockSampler* sampler, ResidualVolume* resVol);
    ~SolveDisplacementKernel();

    void launch() override;
    void solveCPU();

    void pullVertices();

    void forceResidualUpdate();
    void setNumLaunchesBeforeResidualUpdate(unsigned int numLaunches);

    //This returns a pointer to managed memory, use only for debug output to avoid unnecessary cpu/gpu memory paging!
    int3* debugGetImportanceSamplesManaged();
    void debugOutputEquationsCPU();
    void debugOutputEquationsGPU();

    int numBlockOriginsPerIteration;

protected:

    bool canExecute() override;
    void freeCudaResources();

    Solution* solution;
    uint3 solutionDimensions;
    ResidualVolume* residualVolume;
    BlockSampler* sampler;

    Vertex* serializedVertices;
    REAL* serializedMatConfigEquations;

    int3* blockOrigins;
    
    unsigned int numLaunchesSinceLastFullResidualUpdate = UINT_MAX-1; //Trigger a residual update at first iteration
    unsigned int numLaunchesBeforeResidualUpdate = NUM_LAUNCHES_BETWEEN_RESIDUAL_UPDATES;
    FullResidualUpdateKernel fullResidualUpdateKernel;

    void prepareInputs();

    void pushMatConfigEquationsManaged();
    void pushVerticesManaged();
    void allocateBlockOrigins();

    void serializeMaterialConfigurationEquations(void* destination);

    void cpuBuildRHSVector(libmmv::Vec3<REAL>* rhsVec, const MaterialConfigurationEquations* matrices, int x, int y, int z);
    void cpuSolveIteration();
};
