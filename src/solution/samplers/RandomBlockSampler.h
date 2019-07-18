#pragma once
#include "BlockSampler.h"
#include "solution/Solution.h"
#include <random>

class RandomBlockSampler : public BlockSampler {

public:
    RandomBlockSampler(Solution* solution, int blockWorkingAreaSize);
    ~RandomBlockSampler();

    int generateNextBlockOrigins(int3* blockOrigins, int numOriginsToGenerate) override;

private:
    Solution* solution;
    int iteration;
    int blockWorkingSize;
    std::mt19937 rng;

    void writeDebugOutput(int samplingIteration, int3* blockOrigins, int numBlocks);
};