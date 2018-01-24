#include "stdafx.h"
#include <chrono>
#include "RandomBlockSampler.h"
#include "io/VTKSamplingVisualizer.h"

/*
    This sampler generates block origins (lower,left corner of block) in a sequential manner. Origins begin at (0,0,0) and are 
    spaced evenly by the working area size of each block. Once we've gone through all valid locations in the solution the offset 
    is increased by 1 and we start from (offset,offset,offset) again. This ensures the 1-vertex fixed border of each previous block 
    is now inside the working area of the next set of blocks

*/

RandomBlockSampler::RandomBlockSampler(Solution* solution, int blockWorkingAreaSize) :
solution(solution),
blockWorkingSize(blockWorkingAreaSize),
rng(42)
{
    iteration = 0;
}

RandomBlockSampler::~RandomBlockSampler() {

}

int RandomBlockSampler::generateNextBlockOrigins(uint3* blockOrigins, int maxNumBlocks) {
    const libmmv::Vec3ui solutionDims = solution->getSize();
    std::uniform_int_distribution<int> rx(0, solutionDims.x - blockWorkingSize);
    std::uniform_int_distribution<int> ry(0, solutionDims.y - blockWorkingSize);
    std::uniform_int_distribution<int> rz(0, solutionDims.z - blockWorkingSize);

    for (int i = 0; i < maxNumBlocks; i++) {
        blockOrigins[i].x = rx(rng) - 1; //-1 to account for the 1 vertex border which is included in the block origin
        blockOrigins[i].y = ry(rng) - 1;
        blockOrigins[i].z = rz(rng) - 1;
    }

	if (iteration % 200 == 0) {
		//writeDebugOutput(iteration, blockOrigins, maxNumBlocks);
	}
	iteration++;
    return maxNumBlocks;
}

void RandomBlockSampler::writeDebugOutput(int samplingIteration, uint3* blockOrigins, int numBlocks) {
    std::stringstream fp;
    fp << "c:\\tmp\\step_samp_" << samplingIteration << ".vtk";
    VTKSamplingVisualizer vis(solution);
    vis.writeToFile(fp.str(), blockOrigins, numBlocks, blockWorkingSize);
}
