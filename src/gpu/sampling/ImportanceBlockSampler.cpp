#include "stdafx.h"
#include "ImportanceBlockSampler.h"

ImportanceBlockSampler::ImportanceBlockSampler(ResidualVolume* resVol) :
    samplingKernel(resVol)
{
}

ImportanceBlockSampler::~ImportanceBlockSampler()
{
}

int ImportanceBlockSampler::generateNextBlockOrigins(uint3* blockOrigins, int numBlocks)
{
    samplingKernel.setBlockOriginsDestination(blockOrigins);
    samplingKernel.setNumBlocksToFind(numBlocks);
    samplingKernel.launch();
    return numBlocks;
}
