#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>
#include <curand_kernel.h>

#include "CudaKernel.h"
#include "gpu/sampling/ResidualVolume.h"
#include "solution/Vertex.h"

extern "C" void cudaLaunchImportanceSamplingKernel(int3* candidates, const int numCandidatesToFind, const REAL* importancePyramid, const LevelStats* levelStats, curandState* rngStateOnGPU, const int topLevel);
extern "C" void cudaInitializePyramidRNGStates(curandState** rngStateOnGPU, const int numCandidatesToFind);
extern "C" void cudaLaunchPyramidUpdateKernel(REAL* importancePyramid, const int numLevels, const LevelStats* levelStats);

class ImportanceSamplingKernel : public CudaKernel {

public:

    ImportanceSamplingKernel(ResidualVolume* resVol);
    ~ImportanceSamplingKernel();

    void launch() override;
    void setBlockOriginsDestination(int3* blockOrigins);
    void setNumBlocksToFind(int numBlocks);

    REAL getTotalResidual();

protected:

    bool canExecute() override;
    void freeCudaResources();

    ResidualVolume* residualVolume;
    curandState* rngStateOnGPU;
    int3* blockOrigins;
    int numBlocksToGenerate;

    void initCurandState();
};
