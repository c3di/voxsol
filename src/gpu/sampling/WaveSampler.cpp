#include "stdafx.h"
#include "WaveSampler.h"
#include "gpu/GPUParameters.h"

WaveSampler::WaveSampler(Solution * solution, libmmv::Vec3ui& origin, libmmv::Vec3i& direction) :
    solutionSize(solution->getSize())
{
    setWaveOrientation(origin, direction);
    currentWavefrontOrigin = chooseNextWavefrontOrigin();
}

WaveSampler::~WaveSampler()
{
}

int WaveSampler::generateNextBlockOrigins(uint3 * blockOrigins, int numOriginsToGenerate)
{
    for (int i = 0; i < numOriginsToGenerate; i++) {
        blockOrigins[i] = make_uint3(currentWavefrontOrigin.x, currentWavefrontOrigin.y, currentWavefrontOrigin.z);
        nextBlockOrigin(&currentWavefrontOrigin);
    }

    return numOriginsToGenerate;
}

void WaveSampler::progressWavefront() {
    currentWavefront = currentWavefront + waveDirection * BLOCK_SIZE;
    if (currentWavefront.x >= solutionSize.x || currentWavefront.y >= solutionSize.y || currentWavefront.z >= solutionSize.z) {
        moveWavefrontToOrigin();
    }
}

void WaveSampler::moveWavefrontToOrigin() {
    currentWavefront = libmmv::Vec3ui(waveOrigin);

    // In any of these cases the wave is moving in the negative direction through the problem, so the block should start at Edge - BLOCK_SIZE
    if (waveDirection.x < 0) {
        currentWavefront.x += waveDirection.x * (BLOCK_SIZE - 1);
    }
    if (waveDirection.y < 0) {
        currentWavefront.y += waveDirection.y * (BLOCK_SIZE - 1);
    }
    if (waveDirection.z < 0) {
        currentWavefront.z += waveDirection.z * (BLOCK_SIZE - 1);
    }
}

void WaveSampler::nextBlockOrigin(libmmv::Vec3ui* currentWavefrontOrigin) {
    currentWavefrontOrigin->x += BLOCK_SIZE;
    if (currentWavefrontOrigin->x >= solutionSize.x) {
        currentWavefrontOrigin->x = waveOffset.x;
        currentWavefrontOrigin->y += BLOCK_SIZE;
    }
    if (currentWavefrontOrigin->y >= solutionSize.y) {
        currentWavefrontOrigin->y = waveOffset.y;
        currentWavefrontOrigin->z += BLOCK_SIZE;
    }
    if (currentWavefrontOrigin->z >= solutionSize.z) {
        progressWavefront();
        *currentWavefrontOrigin = chooseNextWavefrontOrigin();
    }
}

void WaveSampler::setWaveOrientation(libmmv::Vec3ui& origin, libmmv::Vec3i& direction)
{
    waveOrigin = libmmv::Vec3ui(origin);
    waveDirection = libmmv::Vec3i(direction);
    moveWavefrontToOrigin();
}

libmmv::Vec3ui WaveSampler::chooseNextWavefrontOrigin() {
    std::uniform_int_distribution<int> distribution(-1, 1);
    waveOffset.x = distribution(generator);
    waveOffset.y = distribution(generator);
    waveOffset.z = 0;

    libmmv::Vec3ui wavefront = currentWavefront + waveOffset;
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
