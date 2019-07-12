#pragma once
#include <cuda_runtime.h>

class BlockSampler {
public:
    BlockSampler();
    ~BlockSampler();

    virtual int generateNextBlockOrigins(int3* blockOrigins, int numOriginsToGenerate) = 0;

protected:
    
};