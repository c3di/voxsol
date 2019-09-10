#pragma once
#include "stdafx.h"
#include "libmmv/math/Vec3.h"
#include "problem/DiscreteProblem.h"

struct LevelStats {
    unsigned int startIndex = 0;
    unsigned int sizeX = 0;
    unsigned int sizeY = 0;
    unsigned int sizeZ = 0;
};

class ResidualVolume {
public:
    ResidualVolume(DiscreteProblem* problem);
    ~ResidualVolume();

    void initializePyramidFromProblem();

    REAL* getLocationOfVertexProjectedToLevel(unsigned int level, VertexCoordinate& fullresCoord);
    REAL getResidualOnLevel(unsigned int level, VertexCoordinate& levelCoord) const;
    REAL getResidualOnLevel(unsigned int level ,unsigned int x, unsigned int y, unsigned int z) const;
    void updatePyramid(unsigned int level, VertexCoordinate& from, VertexCoordinate& to);
    void updateEntirePyramid();
    REAL getTotalResidual() const;
    REAL getMaxResidualOnLevelZero() const;
    REAL getAverageResidual(int* numVerticesNotConverged = NULL) const;
    REAL getAverageResidualWithThreshold(REAL threshold, int* numVerticesNotConverged = NULL) const;

    REAL* getPyramidDevicePointer();
    LevelStats* getLevelStatsDevicePointer();
    REAL* getPointerToLevel(unsigned int level);
    LevelStats* getPointerToStatsForLevel(unsigned int level);
    unsigned int getNumberOfLevels() const;

    unsigned int getNumVerticesOnLevelZero() const;

protected:
    // This is GPU managed memory, any read/write operations on it may trigger a costly CPU<->GPU memory sync!
    REAL* importancePyramidManaged = nullptr;
    LevelStats* levelStatsManaged = nullptr;

    DiscreteProblem* problem;

    libmmv::Vec3ui levelZeroSize;
    unsigned int numberOfLevels;

    void freeCudaResources();

    void allocateManagedMemory();
    void initializeLeveLZeroResidualsFromProblem();
    void computeDepthOfPyramid();
};