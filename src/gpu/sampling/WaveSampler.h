#pragma once
#include "solution/samplers/BlockSampler.h"
#include "solution/Solution.h"
#include <random>

class WaveSampler : public BlockSampler {

public:
    WaveSampler(Solution* solution, libmmv::Vec3ui& origin, libmmv::Vec3i& direction);
    ~WaveSampler();

    int generateNextBlockOrigins(uint3* blockOrigins, int numOriginsToGenerate) override;
    

protected:
    libmmv::Vec3ui solutionSize;
    libmmv::Vec3ui waveOrigin;
    libmmv::Vec3i waveDirection;
    libmmv::Vec3ui currentWavefront;
    libmmv::Vec3ui currentWavefrontOrigin;
    libmmv::Vec3i waveOffset;
    unsigned int numBlocksProcessedSinceLastWavefrontReset = 0;
    std::default_random_engine generator;

    libmmv::Vec3ui chooseNextWavefrontOrigin();
    void progressWavefront();
    void nextBlockOrigin(libmmv::Vec3ui* currentWavefrontOrigin);
    void setWaveOrientation(libmmv::Vec3ui& origin, libmmv::Vec3i& direction);
};