#pragma once

#include "libmmv/math/Vec3.h"

class DiscreteProblem;

enum ProblemSide 
{
    POSITIVE_X, POSITIVE_Y, POSITIVE_Z, NEGATIVE_X, NEGATIVE_Y, NEGATIVE_Z
};

class ProblemSideIterator {
public:
    ProblemSideIterator(DiscreteProblem* problem, ProblemSide side );

    libmmv::Vec3ui next();
    bool hasNext();
    libmmv::Vec3i sideIndexToVertexCoordinateOffset(unsigned int sideIndex);

protected:
    DiscreteProblem* problem;
    ProblemSide side;
    libmmv::Vec3ui problemSize;

    unsigned int currentVoxelIndex;

    libmmv::Vec3ui indexToCoordinate(int index);


};
