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
    if (currentWavefront.x >= solutionSize.x) {
        currentWavefront.x = 0;
    }
    if (currentWavefront.y >= solutionSize.y) {
        currentWavefront.y = 0;
    }
    if (currentWavefront.z >= solutionSize.z) {
        currentWavefront.z = 0;
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
    currentWavefront = libmmv::Vec3ui(origin);
}

libmmv::Vec3ui WaveSampler::chooseNextWavefrontOrigin() {
    std::uniform_int_distribution<int> distribution(-1, 1);
    waveOffset.x = distribution(generator);
    waveOffset.y = distribution(generator);
    waveOffset.z = distribution(generator);

    libmmv::Vec3ui wavefront = waveOrigin + waveOffset + (waveDirection * currentWavefront);
    if (wavefront.x > solutionSize.x) {
        wavefront.x = 0;
    }
    if (wavefront.y > solutionSize.y) {
        wavefront.y = 0;
    }
    if (wavefront.z > solutionSize.z) {
        wavefront.z = 0;
    }
    return wavefront;
}
