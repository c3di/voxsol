#include "stdafx.h"
#include "WaveSampler.h"
#include "gpu/GPUParameters.h"

// This sampler tries to lay down update regions in waves that successively get further and further from the origin. 
// The origin should be the side of the problem from which the Neumann force is applied, and the direction should match 
// the direction of the force. For example, if the force is applied in +Z direction the first layer of blocks will be 
// applied at z=0, the next layer will go from z=0 to z=block_size*2, the next from z=0 to z=block_size*3 etc. 
// Once the wave reaches the other side of the problem the process begins again. 

// The goal is to create more frequent updates closer to the Neumann boundaries and to mimic the way the force 
// moves through the problem.

WaveSampler::WaveSampler(Solution * solution, libmmv::Vec3i& origin, libmmv::Vec3i& direction) :
    solutionSize(solution->getSize())
{
    setWaveOrientation(origin, direction);
    currentWaveFront = chooseNextWavefrontOrigin();
}

WaveSampler::~WaveSampler()
{
}

int WaveSampler::generateNextBlockOrigins(int3 * blockOrigins, int numOriginsToGenerate)
{
    for (int i = 0; i < numOriginsToGenerate; i++) {
        blockOrigins[i] = make_int3(currentWaveFront.x, currentWaveFront.y, currentWaveFront.z);
        nextBlockOrigin(&currentWaveFront);
    }

    return numOriginsToGenerate;
}

void WaveSampler::progressWaveEnd() {
    currentWaveEnd = currentWaveEnd + waveDirection * BLOCK_SIZE;
    if (currentWaveEnd.x > solutionSize.x + BLOCK_SIZE || currentWaveEnd.y > solutionSize.y + BLOCK_SIZE || currentWaveEnd.z > solutionSize.z + BLOCK_SIZE) {
        resetWaveEnd();
    }
    if (currentWaveEnd.x <= -BLOCK_SIZE || currentWaveEnd.y <= -BLOCK_SIZE || currentWaveEnd.z <= -BLOCK_SIZE) {
        resetWaveEnd();
    }
}

void WaveSampler::resetWaveEnd() {
    if (waveDirection.x > 0) {
        currentWaveEnd = libmmv::Vec3i(waveDirection.x * BLOCK_SIZE, solutionSize.y, solutionSize.z);
    }
    else if (waveDirection.x < 0) {
        currentWaveEnd = libmmv::Vec3i(solutionSize.x + waveDirection.x * BLOCK_SIZE, solutionSize.y, solutionSize.z);
    } 
    else if (waveDirection.y > 0) {
        currentWaveEnd = libmmv::Vec3i(solutionSize.x, waveDirection.y * BLOCK_SIZE, solutionSize.z);
    }
    else if (waveDirection.y < 0) {
        currentWaveEnd = libmmv::Vec3i(solutionSize.x, solutionSize.y + waveDirection.y * BLOCK_SIZE, solutionSize.z);
    }
    else if (waveDirection.z > 0) {
        currentWaveEnd = libmmv::Vec3i(solutionSize.x, solutionSize.y, waveDirection.z * BLOCK_SIZE);
    }
    else if (waveDirection.z < 0) {
        currentWaveEnd = libmmv::Vec3i(solutionSize.x, solutionSize.y, solutionSize.z + waveDirection.z * BLOCK_SIZE);
    }
}

void WaveSampler::nextBlockOrigin(libmmv::Vec3i* currentWaveFront) {
    if (waveDirection.z > 0) {
        currentWaveFront->x += BLOCK_SIZE;
        if (currentWaveFront->x > currentWaveEnd.x) {
            currentWaveFront->x = waveOffset.x;
            currentWaveFront->y += BLOCK_SIZE;
        }
        if (currentWaveFront->y > currentWaveEnd.y) {
            currentWaveFront->y = waveOffset.y;
            currentWaveFront->z += BLOCK_SIZE;
        }
        if (currentWaveFront->z > currentWaveEnd.z) {
            progressWaveEnd();
            *currentWaveFront = chooseNextWavefrontOrigin();
        }
        return;
    }

    if (waveDirection.z < 0) {
        currentWaveFront->x += BLOCK_SIZE;
        if (currentWaveFront->x > currentWaveEnd.x) {
            currentWaveFront->x = waveOffset.x;
            currentWaveFront->y += BLOCK_SIZE;
        }
        if (currentWaveFront->y > currentWaveEnd.y) {
            currentWaveFront->y = waveOffset.y;
            currentWaveFront->z -= BLOCK_SIZE;
        }
        if (currentWaveFront->z < currentWaveEnd.z) {
            progressWaveEnd();
            *currentWaveFront = chooseNextWavefrontOrigin();
        }
        return;
    }

}

void WaveSampler::setWaveOrientation(libmmv::Vec3i& origin, libmmv::Vec3i& direction)
{
    waveOrigin = libmmv::Vec3ui(origin);
    waveDirection = libmmv::Vec3i(direction);
    resetWaveEnd();
}

libmmv::Vec3i WaveSampler::chooseNextWavefrontOrigin() {
    std::uniform_int_distribution<int> distribution(-1, 1);
    waveOffset.x = distribution(generator);
    waveOffset.y = distribution(generator);
    waveOffset.z = distribution(generator);

    libmmv::Vec3i wavefront = waveOrigin + waveOffset;
    if (wavefront.x > solutionSize.x) {
        wavefront.x = waveOrigin.x;
    }
    if (wavefront.y > solutionSize.y) {
        wavefront.y = waveOrigin.y;
    }
    if (wavefront.z > solutionSize.z) {
        wavefront.z = waveOrigin.z;
    }
    return wavefront;
}
