#pragma once
#include "solution/samplers/BlockSampler.h"
#include "solution/Solution.h"
#include <random>

class WaveSampler : public BlockSampler {

public:
    WaveSampler(Solution* solution, libmmv::Vec3i& origin, libmmv::Vec3i& direction);
    ~WaveSampler();

    int generateNextBlockOrigins(int3* blockOrigins, int numOriginsToGenerate) override;
    

protected:
    libmmv::Vec3i solutionSize;
    libmmv::Vec3i waveOrigin;
    libmmv::Vec3i waveDirection;
    libmmv::Vec3i currentWaveFront;
    libmmv::Vec3i currentWaveEnd;
    libmmv::Vec3i waveOffset;
    unsigned int numBlocksProcessedSinceLastWavefrontReset = 0;

    std::default_random_engine generator;

    libmmv::Vec3i chooseNextWavefrontOrigin();
    void progressWaveEnd();
    void nextBlockOrigin(libmmv::Vec3i* currentWavefrontOrigin);
    void setWaveOrientation(libmmv::Vec3i& origin, libmmv::Vec3i& direction);
    void resetWaveEnd();
};