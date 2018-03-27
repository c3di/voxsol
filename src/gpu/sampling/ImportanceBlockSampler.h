#pragma once
#include "solution/samplers/BlockSampler.h"
#include "solution/Solution.h"
#include "ResidualVolume.h"
#include "gpu/kernels/ImportanceSamplingKernel.h"
#include <random>

class ImportanceBlockSampler : public BlockSampler {

public:
    ImportanceBlockSampler(ResidualVolume* resVol);
    ~ImportanceBlockSampler();

    int generateNextBlockOrigins(uint3* blockOrigins, int numOriginsToGenerate) override;

protected:
    ImportanceSamplingKernel samplingKernel;
};