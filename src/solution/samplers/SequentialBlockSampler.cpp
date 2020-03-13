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
rng()
{
    lastOrigin.x = lastOrigin.y = lastOrigin.z = 0;
    currentOffset.x = currentOffset.y = currentOffset.z = 0;
    iteration = 0;
}

SequentialBlockSampler::~SequentialBlockSampler() {

}

int SequentialBlockSampler::generateNextBlockOrigins(int3* blockOrigins, int maxNumBlocks) {
    const libmmv::Vec3ui solutionDims = solution->getSize();
    int i;
    int halfBlockSize = (blockStride - 1) / 2;

    for (i = 0; i < maxNumBlocks; i++) {
        if (lastOrigin.x > (int)solutionDims.x) {
            lastOrigin.x = currentOffset.x;
            lastOrigin.y += blockStride;
        }

        if (lastOrigin.y > (int)solutionDims.y) {
            lastOrigin.y = currentOffset.y;
            lastOrigin.z += blockStride;
        }

        if (lastOrigin.z > (int)solutionDims.z) {
            // Note: It's important to add a random offset to the block origins, otherwise the simulation can produce 
            // self-reinforcing errors that eventually lead to divergence
            shiftOffsetRandomly();
            lastOrigin.x = currentOffset.x;
            lastOrigin.y = currentOffset.y;
            lastOrigin.z = currentOffset.z;
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
        blockOrigins[j].x = INT_MAX;
        blockOrigins[j].y = INT_MAX;
        blockOrigins[j].z = INT_MAX;
    }

   // writeDebugOutput(iteration, blockOrigins, i);
    iteration++;

    return i;
}

void SequentialBlockSampler::shiftOffsetDeterministically() {
    if (iteration % 2 == 0) {
        currentOffset.x = 0;
        currentOffset.y = 0;
        //currentOffset.z += 1;
    }
    else {
        currentOffset.x = -1;
        currentOffset.y = -1;
        //currentOffset.z -= 1;
    }
    
}

void SequentialBlockSampler::shiftOffsetRandomly() {
    std::random_device rd;
    std::uniform_int_distribution<int> rngOffset(-1, 1);

    currentOffset.x = rngOffset(rng);
    currentOffset.y = rngOffset(rng);
    currentOffset.z = rngOffset(rng);
}

void SequentialBlockSampler::writeDebugOutput(int samplingIteration, int3* blockOrigins, int numBlocks) {
    std::stringstream fp;
    fp << "c:\\tmp\\step_samp_" << samplingIteration << ".vtk";
    VTKSamplingVisualizer vis(solution);
    vis.writeToFile(fp.str(), blockOrigins, numBlocks, blockStride - 1);
}
