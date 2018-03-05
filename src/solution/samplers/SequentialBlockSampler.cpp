#include "stdafx.h"
#include <climits>
#include "SequentialBlockSampler.h"
#include "io/VTKSamplingVisualizer.h"

/*
    This sampler generates block origins (lower,left corner of block) in a sequential manner. Origins begin at (0,0,0) and are 
    spaced evenly by the working area size of each block. Once we've gone through all valid locations in the solution the offset 
    is increased by 1 and we start from (offset,offset,offset) again. This ensures the 1-vertex fixed border of each previous block 
    is now inside the working area of the next set of blocks

*/

SequentialBlockSampler::SequentialBlockSampler(Solution* solution, int blockWorkingAreaSize) : 
solution(solution),
blockStride(blockWorkingAreaSize), 
rng(42)
{
    lastOrigin.x = lastOrigin.y = lastOrigin.z = 0;
    currentOffset.x = currentOffset.y = currentOffset.z = 0;
    iteration = 0;
}

SequentialBlockSampler::~SequentialBlockSampler() {

}

int SequentialBlockSampler::generateNextBlockOrigins(uint3* blockOrigins, int maxNumBlocks) {
    const libmmv::Vec3ui solutionDims = solution->getSize();
    int i;
    int halfBlockSize = (blockStride - 1) / 2;
    for (i = 0; i < maxNumBlocks; i++) {
#pragma warning(suppress: 4018) //Suppress signed/unsigned mismatch in conditional
        if (lastOrigin.x >= solutionDims.x - blockStride) {
            lastOrigin.x = currentOffset.x;
            lastOrigin.y += blockStride;
        }
#pragma warning(suppress: 4018)
        if (lastOrigin.y >= solutionDims.y - blockStride) {
            lastOrigin.y = currentOffset.y;
            lastOrigin.z += blockStride;
        }
#pragma warning(suppress: 4018)
        if (lastOrigin.z >= solutionDims.z - blockStride) {
            shiftOffsetDeterministically();
            lastOrigin.x = currentOffset.x;
            lastOrigin.y = currentOffset.y;
            lastOrigin.z = currentOffset.z;
            break; //Don't generate any more blocks to ensure no blocks overlap (can cause divergence)
        }

        // -1 to account for the 1-vertex border of fixed vertices. We want to choose origins for the working area but
        // the blocks themselves are 1 vertex bigger in each dimension, so the origin needs to be shifted by 1
        blockOrigins[i].x = lastOrigin.x;
        blockOrigins[i].y = lastOrigin.y;
        blockOrigins[i].z = lastOrigin.z;

        lastOrigin.x += blockStride;
    }

    // Invalidate remaining blocks so they aren't processed during the displacement update phase
    for (int j = i; j < maxNumBlocks; j++) {
        blockOrigins[j].x = UINT_MAX;
        blockOrigins[j].y = UINT_MAX;
        blockOrigins[j].z = UINT_MAX;
    }

   // writeDebugOutput(iteration, blockOrigins, i);
    iteration++;

    return i;
}

void SequentialBlockSampler::shiftOffsetDeterministically() {
    if (iteration % 2 == 0) {
        currentOffset.x += 1;
        currentOffset.y += 1;
        currentOffset.z += 1;
    }
    else {
        currentOffset.x -= 1;
        currentOffset.y -= 1;
        currentOffset.z -= 1;
    }
    
}

void SequentialBlockSampler::shiftOffsetStochastically() {
    std::random_device rd;
    std::uniform_int_distribution<int> rngOffset(0, blockStride / 2);

    currentOffset.x = rngOffset(rng);
    currentOffset.y = rngOffset(rng);
    currentOffset.z = rngOffset(rng);
}

void SequentialBlockSampler::writeDebugOutput(int samplingIteration, uint3* blockOrigins, int numBlocks) {
    std::stringstream fp;
    fp << "c:\\tmp\\step_samp_" << samplingIteration << ".vtk";
    VTKSamplingVisualizer vis(solution);
    vis.writeToFile(fp.str(), blockOrigins, numBlocks, blockStride - 1);
}
