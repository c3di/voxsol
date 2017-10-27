#pragma once
#include "BlockSampler.h"
#include "solution/Solution.h"
#include <random>

class SequentialBlockSampler : public BlockSampler {

public:
    SequentialBlockSampler(Solution* solution, int blockWorkingAreaSize);
    ~SequentialBlockSampler();

    int generateNextBlockOrigins(int3* blockOrigins, int numOriginsToGenerate) override;

private:
    Solution* solution;
    const int blockStride;
    int3 lastOrigin;
    int3 currentOffset;
    int iteration;
    std::mt19937 rng;

    void shiftOffsetStochastically();
    void writeDebugOutput(int samplingIteration, int3* blockOrigins, int numBlocks);
};