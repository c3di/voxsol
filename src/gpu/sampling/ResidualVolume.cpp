#include "stdafx.h"
#include "ResidualVolume.h"
#include "problem/DiscreteProblem.h"
#include "gpu/CudaCommonFunctions.h"

ResidualVolume::ResidualVolume(DiscreteProblem* problem) :
    problem(problem),
    importancePyramidManaged(nullptr),
    levelStatsManaged(nullptr)
{

}

ResidualVolume::~ResidualVolume()
{
    freeCudaResources();
}


void ResidualVolume::freeCudaResources() {
    if (importancePyramidManaged != nullptr) {
        cudaCheckSuccess(cudaFree(importancePyramidManaged));
        importancePyramidManaged = nullptr;
    }
    if (levelStatsManaged != nullptr) {
        cudaCheckSuccess(cudaFree(levelStatsManaged));
        levelStatsManaged = nullptr;
    }
}

REAL* ResidualVolume::getLocationOfVertexProjectedToLevel(unsigned int level, VertexCoordinate& fullresCoord) {
    unsigned int x = fullresCoord.x >> (level+1); //divide by 2 level+1 times, +1 because level zero is already fullres/2
    unsigned int y = fullresCoord.y >> (level+1);
    unsigned int z = fullresCoord.z >> (level+1);
    LevelStats* levelStats = &levelStatsManaged[level];
    REAL* projectedLocation = importancePyramidManaged + levelStats->startIndex;

    projectedLocation += z * levelStats->sizeX * levelStats->sizeY + y * levelStats->sizeX + x;
    return projectedLocation;
}

REAL ResidualVolume::getResidualOnLevel(unsigned int level, VertexCoordinate & levelCoord) const
{
    return getResidualOnLevel(level, levelCoord.x, levelCoord.y, levelCoord.z);
}

REAL ResidualVolume::getMaxResidualOnLevelZero() const {
    LevelStats levelZeroStats = levelStatsManaged[0];
    REAL* levelStart = &importancePyramidManaged[levelZeroStats.startIndex];
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

REAL ResidualVolume::getAverageResidual(int* numVerticesNotConverged) const {
    LevelStats levelZeroStats = levelStatsManaged[0];
    REAL* levelStart = &importancePyramidManaged[levelZeroStats.startIndex];
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

REAL ResidualVolume::getAverageResidualWithThreshold(REAL ignoreThreshold, int* numVerticesNotConverged) const {
    LevelStats levelZeroStats = levelStatsManaged[0];
    REAL* levelStart = &importancePyramidManaged[levelZeroStats.startIndex];
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

REAL ResidualVolume::getResidualOnLevel(unsigned int level, unsigned int x, unsigned int y, unsigned int z) const
{
    LevelStats statsForLevel = levelStatsManaged[level];

    if (x < 0 || x >= statsForLevel.sizeX)
        return asREAL(0.0);
    if (y < 0 || y >= statsForLevel.sizeY)
        return asREAL(0.0);
    if (z < 0 || z >= statsForLevel.sizeZ)
        return asREAL(0.0);

    REAL* levelPointer = importancePyramidManaged + statsForLevel.startIndex;
    levelPointer += z * statsForLevel.sizeY * statsForLevel.sizeX + y * statsForLevel.sizeX + x;
    return *levelPointer;
}

REAL ResidualVolume::getTotalResidual() const {
    return getResidualOnLevel(numberOfLevels-1, 0, 0, 0);
}

void ResidualVolume::updatePyramid(unsigned int level, VertexCoordinate & from, VertexCoordinate & to) {
    if (level >= numberOfLevels) {
        return;
    }
    if (level != 0) {
        LevelStats statsForLevel = levelStatsManaged[level];
        REAL* levelStart = &importancePyramidManaged[statsForLevel.startIndex];

        for (unsigned int z = from.z; z <= to.z; z++)
            for (unsigned int y = from.y; y <= to.y; y++)
                for (unsigned int x = from.x; x <= to.x; x++) {
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
    REAL* p = importancePyramidManaged + levelStatsManaged[level].startIndex;
    return p;
}

REAL* ResidualVolume::getPyramidDevicePointer()
{
    return importancePyramidManaged;
}

LevelStats* ResidualVolume::getLevelStatsDevicePointer()
{
    return levelStatsManaged;
}

LevelStats* ResidualVolume::getPointerToStatsForLevel(unsigned int level)
{
    LevelStats* l = levelStatsManaged + level;
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
    allocateManagedMemory();
    initializeLeveLZeroResidualsFromProblem();
    updateEntirePyramid();
}

void ResidualVolume::computeDepthOfPyramid()
{
    levelZeroSize = (problem->getSize() + libmmv::Vec3ui(2, 2, 2)) / 2;
    unsigned int maxDim = std::max(std::max(levelZeroSize.x, levelZeroSize.y), levelZeroSize.z);
    double log = std::log((double)maxDim) / std::log(2.0);
    numberOfLevels = (int)std::ceil(log) + 1;
}

void ResidualVolume::allocateManagedMemory() {
    unsigned int x = problem->getSize().x + 1; // number of vertices is 1 greater than number of voxels
    unsigned int y = problem->getSize().y + 1;
    unsigned int z = problem->getSize().z + 1;

    cudaCheckSuccess(cudaMallocManaged(&levelStatsManaged, numberOfLevels * sizeof(LevelStats)));

    size_t totalSizeInBytes = 0;
    unsigned int levelIndex = 0;
    for (unsigned int i = 0; i < numberOfLevels; i++) {
        if (i > 0) {
            // Next level starts at the end of the previous level, except level 0 which starts at 0
            levelIndex += x*y*z;
            levelStatsManaged[i].startIndex = levelIndex;
        }
        x = (x+1) / 2; y = (y+1) / 2; z = (z+1) / 2;
        totalSizeInBytes += x * y * z * sizeof(REAL);
        levelStatsManaged[i].sizeX = x;
        levelStatsManaged[i].sizeY = y;
        levelStatsManaged[i].sizeZ = z;
    }

    cudaCheckSuccess(cudaMallocManaged(&importancePyramidManaged, totalSizeInBytes));
    memset(importancePyramidManaged, 0, totalSizeInBytes);
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


