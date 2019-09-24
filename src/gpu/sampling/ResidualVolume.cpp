#include "stdafx.h"
#include "ResidualVolume.h"
#include "problem/DiscreteProblem.h"
#include "gpu/CudaCommonFunctions.h"

ResidualVolume::ResidualVolume(DiscreteProblem* problem) :
    problem(problem)
{
    for (int i = 0; i < NUMBER_OF_BUFFERS; i++) {
        residualPyramidBuffers[i] = nullptr;
        levelStatsBuffers[i] = nullptr;
    }
}

ResidualVolume::~ResidualVolume()
{
    freeCudaResources();
}


void ResidualVolume::freeCudaResources() {
    for (int i = 0; i < NUMBER_OF_BUFFERS; i++) {
        if (residualPyramidBuffers[i] != nullptr) {
            cudaCheckSuccess(cudaFree(residualPyramidBuffers[i]));
            residualPyramidBuffers[i] = nullptr;
        }
        if (levelStatsBuffers[i] != nullptr) {
            cudaCheckSuccess(cudaFree(levelStatsBuffers[i]));
            levelStatsBuffers[i] = nullptr;
        }
    }
}

REAL* ResidualVolume::getLocationOfVertexProjectedToLevel(unsigned int level, VertexCoordinate& fullresCoord) {
    unsigned int x = fullresCoord.x >> (level+1); //divide by 2 level+1 times, +1 because level zero is already fullres/2
    unsigned int y = fullresCoord.y >> (level+1);
    unsigned int z = fullresCoord.z >> (level+1);
    LevelStats* levelStats = &getActiveLevelStatsObject()[level];
    REAL* projectedLocation = getActiveResidualBuffer() + levelStats->startIndex;

    projectedLocation += z * levelStats->sizeX * levelStats->sizeY + y * levelStats->sizeX + x;
    return projectedLocation;
}

REAL ResidualVolume::getResidualOnLevel(unsigned int level, VertexCoordinate & levelCoord)
{
    return getResidualOnLevel(level, levelCoord.x, levelCoord.y, levelCoord.z);
}

REAL ResidualVolume::getMaxResidualOnLevelZero() {
    LevelStats levelZeroStats = getActiveLevelStatsObject()[0];
    REAL* levelStart = &getActiveResidualBuffer()[levelZeroStats.startIndex];
    REAL maxResidual = 0;

    for (unsigned int z = 0; z <= levelZeroStats.sizeZ; z++)
        for (unsigned int y = 0; y <= levelZeroStats.sizeY; y++)
            for (unsigned int x = 0; x <= levelZeroStats.sizeX; x++) {
                REAL res = getResidualOnLevel(0, x, y, z);
                if (res > maxResidual) {
                    maxResidual = res;
                }
            }

    return maxResidual;
}

REAL ResidualVolume::getResidualDeltaToLastUpdate(int* numVerticesNotConverged) {
    LevelStats levelZeroStats = getActiveLevelStatsObject()[0];
    int levelZeroNumEntries = levelZeroStats.sizeX * levelZeroStats.sizeY * levelZeroStats.sizeZ;

    REAL* lastBuffer;
    if (currentActiveBuffer == 0) {
        lastBuffer = residualPyramidBuffers[NUMBER_OF_BUFFERS - 1];
    }
    else {
        lastBuffer = residualPyramidBuffers[currentActiveBuffer - 1];
    }

    REAL* currentBuffer = getActiveResidualBuffer();
    int endOfLevelZero = levelZeroStats.startIndex + levelZeroNumEntries;

    REAL totalDelta = 0;
    int totalVerticesNotZero = 0;

    for (int i = levelZeroStats.startIndex; i < endOfLevelZero; i++) {
        if (currentBuffer[i] != 0) {
            totalDelta += abs(lastBuffer[i] - currentBuffer[i]);
            totalVerticesNotZero++;
        }
    }

    if (numVerticesNotConverged != nullptr) {
        *numVerticesNotConverged = totalVerticesNotZero;
    }

    if (totalVerticesNotZero == 0) {
        return 0;
    }

    return totalDelta / totalVerticesNotZero;
}

REAL ResidualVolume::getAverageResidual(int* numVerticesNotConverged) {
    LevelStats levelZeroStats = getActiveLevelStatsObject()[0];
    REAL* levelStart = &getActiveResidualBuffer()[levelZeroStats.startIndex];
    double totalResidual = 0;
    int numResidualsGreaterThanEps = 0;

    for (unsigned int z = 0; z < levelZeroStats.sizeZ; z++)
        for (unsigned int y = 0; y < levelZeroStats.sizeY; y++)
            for (unsigned int x = 0; x < levelZeroStats.sizeX; x++) {
                double res = 1.0 + getResidualOnLevel(0, x, y, z);
                if (res > 1.0) {
                    totalResidual += res * res;
                    numResidualsGreaterThanEps++;
                }
            }
    if (numVerticesNotConverged != NULL) {
        *numVerticesNotConverged = numResidualsGreaterThanEps;
    }

    return asREAL( (totalResidual / (double) numResidualsGreaterThanEps) - 1.0 );
}

REAL ResidualVolume::getAverageResidualWithThreshold(REAL ignoreThreshold, int* numVerticesNotConverged) {
    LevelStats levelZeroStats = getActiveLevelStatsObject()[0];
    REAL* levelStart = &getActiveResidualBuffer()[levelZeroStats.startIndex];
    double totalResidual = 0;
    int numResidualsGreaterThanEps = 0;

    for (unsigned int z = 0; z < levelZeroStats.sizeZ; z++)
        for (unsigned int y = 0; y < levelZeroStats.sizeY; y++)
            for (unsigned int x = 0; x < levelZeroStats.sizeX; x++) {
                double res = getResidualOnLevel(0, x, y, z);
                if (res > ignoreThreshold) {
                    totalResidual += res;
                    numResidualsGreaterThanEps++;
                }
            }
    if (numVerticesNotConverged != NULL) {
        *numVerticesNotConverged = numResidualsGreaterThanEps;
    }

    if (numResidualsGreaterThanEps == 0) {
        return ignoreThreshold;
    }

    return asREAL((totalResidual / (double)numResidualsGreaterThanEps));
}

REAL ResidualVolume::getResidualOnLevel(unsigned int level, unsigned int x, unsigned int y, unsigned int z)
{
    LevelStats statsForLevel = getActiveLevelStatsObject()[level];

    if (x < 0 || x >= statsForLevel.sizeX)
        return asREAL(0.0);
    if (y < 0 || y >= statsForLevel.sizeY)
        return asREAL(0.0);
    if (z < 0 || z >= statsForLevel.sizeZ)
        return asREAL(0.0);

    REAL* levelPointer = getActiveResidualBuffer() + statsForLevel.startIndex;
    levelPointer += z * statsForLevel.sizeY * statsForLevel.sizeX + y * statsForLevel.sizeX + x;
    return *levelPointer;
}

REAL ResidualVolume::getResidualDeltaOnLevel(unsigned int level, unsigned int x, unsigned int y, unsigned int z)
{
    LevelStats statsForLevel = getActiveLevelStatsObject()[level];
    int locationInBuffer = statsForLevel.startIndex + statsForLevel.sizeX * statsForLevel.sizeY * z + statsForLevel.sizeX * y + x;

    if (x < 0 || x >= statsForLevel.sizeX)
        return asREAL(0.0);
    if (y < 0 || y >= statsForLevel.sizeY)
        return asREAL(0.0);
    if (z < 0 || z >= statsForLevel.sizeZ)
        return asREAL(0.0);

    REAL* lastBuffer;
    if (currentActiveBuffer == 0) {
        lastBuffer = residualPyramidBuffers[NUMBER_OF_BUFFERS - 1];
    }
    else {
        lastBuffer = residualPyramidBuffers[currentActiveBuffer - 1];
    }

    REAL* currentBuffer = getActiveResidualBuffer();

    return abs(lastBuffer[locationInBuffer] - currentBuffer[locationInBuffer]);

}

REAL ResidualVolume::getTotalResidual() {
    return getResidualOnLevel(numberOfLevels-1, 0, 0, 0);
}

void ResidualVolume::updatePyramid(unsigned int level, VertexCoordinate & from, VertexCoordinate & to) {
    if (level >= numberOfLevels) {
        return;
    }
    if (level != 0) {
        LevelStats statsForLevel = getActiveLevelStatsObject()[level];
        REAL* levelStart = &getActiveResidualBuffer()[statsForLevel.startIndex];

        for (unsigned int z = from.z; z <= to.z; z++)
            for (unsigned int y = from.y; y <= to.y; y++)
                for (unsigned int x = from.x; x <= to.x; x++) {
                    if (x >= statsForLevel.sizeX || y >= statsForLevel.sizeY || z >= statsForLevel.sizeZ) {
                        continue;
                    }
                    REAL* currentEntry = levelStart + z * statsForLevel.sizeX * statsForLevel.sizeY + y * statsForLevel.sizeX + x;
                    *currentEntry = asREAL(0.0);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x,     2 * y,     2 * z);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x + 1, 2 * y,     2 * z);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x,     2 * y + 1, 2 * z);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x + 1, 2 * y + 1, 2 * z);

                    *currentEntry += getResidualOnLevel(level - 1, 2 * x,     2 * y,     2 * z + 1);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x + 1, 2 * y,     2 * z + 1);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x,     2 * y + 1, 2 * z + 1);
                    *currentEntry += getResidualOnLevel(level - 1, 2 * x + 1, 2 * y + 1, 2 * z + 1);
                    currentEntry++;
                }
    }
    updatePyramid(level + 1, from / 2, to / 2);
}

void ResidualVolume::updateEntirePyramid() {
    return updatePyramid(0, VertexCoordinate(0, 0, 0), levelZeroSize);
}

REAL* ResidualVolume::getPointerToLevel(unsigned int level)
{
    REAL* p = getActiveResidualBuffer() + getActiveLevelStatsObject()[level].startIndex;
    return p;
}

REAL* ResidualVolume::getActiveResidualBuffer()
{
    return residualPyramidBuffers[currentActiveBuffer];
}

REAL* ResidualVolume::getNextBufferForResidualUpdate()
{
    currentActiveBuffer = (currentActiveBuffer + 1) % NUMBER_OF_BUFFERS;
    return residualPyramidBuffers[currentActiveBuffer];
}

LevelStats* ResidualVolume::getActiveLevelStatsObject()
{
    return levelStatsBuffers[currentActiveBuffer];
}

LevelStats* ResidualVolume::getPointerToStatsForLevel(unsigned int level)
{
    LevelStats* l = getActiveLevelStatsObject() + level;
    return l;
}

unsigned int ResidualVolume::getNumberOfLevels() const {
    return numberOfLevels;
}

unsigned int ResidualVolume::getNumVerticesOnLevelZero() const
{
    return levelZeroSize.x * levelZeroSize.y * levelZeroSize.z;
}

void ResidualVolume::initializePyramidFromProblem() {
    computeDepthOfPyramid();

    for (int i = 0; i < NUMBER_OF_BUFFERS; i++) {
        allocateManagedMemory(i);
    }

    updateEntirePyramid();
}

void ResidualVolume::computeDepthOfPyramid()
{
    levelZeroSize = (problem->getSize() + libmmv::Vec3ui(2, 2, 2)) / 2;
    unsigned int maxDim = std::max(std::max(levelZeroSize.x, levelZeroSize.y), levelZeroSize.z);
    double log = std::log((double)maxDim) / std::log(2.0);
    numberOfLevels = (int)std::ceil(log) + 1;
}

void ResidualVolume::allocateManagedMemory(int bufferNumber) {
    unsigned int x = problem->getSize().x + 1; // number of vertices is 1 greater than number of voxels
    unsigned int y = problem->getSize().y + 1;
    unsigned int z = problem->getSize().z + 1;

    cudaCheckSuccess(cudaMallocManaged(&levelStatsBuffers[bufferNumber], numberOfLevels * sizeof(LevelStats)));

    LevelStats* levelStats = levelStatsBuffers[bufferNumber];
    levelStats[0].startIndex = 0;

    size_t totalSizeInBytes = 0;
    unsigned int levelIndex = 0;
    for (unsigned int i = 0; i < numberOfLevels; i++) {
        if (i > 0) {
            // Next level starts at the end of the previous level, except level 0 which starts at 0
            levelIndex += x*y*z;
            levelStats[i].startIndex = levelIndex;
        }
        x = (x+1) / 2; y = (y+1) / 2; z = (z+1) / 2;
        totalSizeInBytes += x * y * z * sizeof(REAL);
        levelStats[i].sizeX = x;
        levelStats[i].sizeY = y;
        levelStats[i].sizeZ = z;
    }

    cudaCheckSuccess(cudaMallocManaged(&residualPyramidBuffers[bufferNumber], totalSizeInBytes));
    memset(residualPyramidBuffers[bufferNumber], 0, totalSizeInBytes);
}

// Initially all residuals are 0 and the only "active" region of the simulation are the Neumann Boundary areas,
// so we set those areas to 1 to ensure the first simulation pass updates them first
void ResidualVolume::initializeLeveLZeroResidualsFromProblem() {
    std::unordered_map<unsigned int, NeumannBoundary>* neumannBoundaries = problem->getNeumannBoundaryMap();

    for (auto it = neumannBoundaries->begin(); it != neumannBoundaries->end(); it++) {
        VertexCoordinate vertexCoord = problem->mapToVertexCoordinate(it->first);
        REAL* locationInPyramid = getLocationOfVertexProjectedToLevel(0, vertexCoord);
        *locationInPyramid += asREAL(1.0);
    }
}


