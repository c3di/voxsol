#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "gpu/sampling/ImportanceVolume.h"
#include "solution/Vertex.h"

extern "C" void cudaLaunchImportanceSamplingKernel(uint3* candidates, const int numCandidatesToFind, const REAL* importancePyramid, const LevelStats* levelStats, const int topLevel);

class ImportanceSamplingKernel : public CudaKernel {

public:

    ImportanceSamplingKernel(ImportanceVolume* impVol, unsigned int numCandidates = 1024);
    ~ImportanceSamplingKernel();

    void launch() override;

    void setNumberOfCandidatesToFind(unsigned int numCandidates);
    uint3* getBlockOriginsDevicePointer();

protected:

    bool canExecute() override;
    void freeCudaResources();

    ImportanceVolume* importanceVolume;
    uint3* blockOrigins;
    unsigned int numberOfCandidatesToFind;

    void allocateCandidatesArray();
};
