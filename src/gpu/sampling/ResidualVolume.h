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

#define NUMBER_OF_BUFFERS 2

class ResidualVolume {
public:
    ResidualVolume(DiscreteProblem* problem);
    ~ResidualVolume();

    void initializePyramidFromProblem();

    REAL* getLocationOfVertexProjectedToLevel(unsigned int level, VertexCoordinate& fullresCoord);
    REAL getResidualOnLevel(unsigned int level, VertexCoordinate& levelCoord);
    REAL getResidualOnLevel(unsigned int level ,unsigned int x, unsigned int y, unsigned int z) ;
    void updatePyramid(unsigned int level, VertexCoordinate& from, VertexCoordinate& to);
    void updateEntirePyramid();
    REAL getTotalResidual() ;
    REAL getMaxResidualOnLevelZero() ;
    REAL getAverageResidual(int* numVerticesNotConverged = NULL) ;
    REAL getAverageResidualWithThreshold(REAL threshold, int* numVerticesNotConverged = NULL) ;

    REAL getResidualDeltaToLastUpdate(int* numVerticesNotConverged);
    REAL getResidualDeltaOnLevel(unsigned int level, unsigned int x, unsigned int y, unsigned int z);

    REAL* getActiveResidualBuffer() ;
    REAL* getNextBufferForResidualUpdate();

    LevelStats* getActiveLevelStatsObject();

    REAL* getPointerToLevel(unsigned int level);
    LevelStats* getPointerToStatsForLevel(unsigned int level);
    
    unsigned int getNumberOfLevels() const;
    unsigned int getNumVerticesOnLevelZero() const;

protected:
    // This is GPU managed memory, any read/write operations on it may trigger a costly CPU<->GPU memory sync!
    REAL* residualPyramidBuffers[NUMBER_OF_BUFFERS];
    LevelStats* levelStatsBuffers[NUMBER_OF_BUFFERS];
    int currentActiveBuffer = 0;

    DiscreteProblem* problem;

    libmmv::Vec3ui levelZeroSize;
    unsigned int numberOfLevels;

    void freeCudaResources();

    void allocateManagedMemory(int bufferNumber);
    void initializeLeveLZeroResidualsFromProblem();
    void computeDepthOfPyramid();
};