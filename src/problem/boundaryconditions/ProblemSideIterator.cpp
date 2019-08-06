#include "stdafx.h"

#include "ProblemSideIterator.h"
#include "problem/DiscreteProblem.h"

// Used to enumerate the vertices of a particular voxel face, counter-clockwise from lower left corner to upper left corner
const libmmv::Vec3i offsetsPosX[4] = { libmmv::Vec3i(0, 0, 0), libmmv::Vec3i(0, 0, 1), libmmv::Vec3i(0, 1, 1), libmmv::Vec3i(0, 1, 0) };
const libmmv::Vec3i offsetsPosY[4] = { libmmv::Vec3i(0, 0, 0), libmmv::Vec3i(1, 0, 0), libmmv::Vec3i(1, 0, 1), libmmv::Vec3i(0, 0, 1) };
const libmmv::Vec3i offsetsPosZ[4] = { libmmv::Vec3i(0, 0, 0), libmmv::Vec3i(1, 0, 0), libmmv::Vec3i(1, 1, 0), libmmv::Vec3i(0, 1, 0) };

// When projecting from the negative direction we need to offset by one in the projection direction also to target the vertices 
// on the other side of the voxel (the face closest to the projection direction)
const libmmv::Vec3i offsetsNegX[4] = { libmmv::Vec3i(1, 0, 0), libmmv::Vec3i(1, 1, 0), libmmv::Vec3i(1, 1, 1), libmmv::Vec3i(1, 0, 1) };
const libmmv::Vec3i offsetsNegY[4] = { libmmv::Vec3i(0, 1, 0), libmmv::Vec3i(1, 1, 0), libmmv::Vec3i(1, 1, 1), libmmv::Vec3i(0, 1, 1) };
const libmmv::Vec3i offsetsNegZ[4] = { libmmv::Vec3i(0, 0, 1), libmmv::Vec3i(1, 0, 1), libmmv::Vec3i(1, 1, 1), libmmv::Vec3i(0, 1, 1) };



ProblemSideIterator::ProblemSideIterator(DiscreteProblem * problem, ProblemSide side)
    : problem(problem), side(side), problemSize(problem->getSize()), currentVoxelIndex(0)
{
    
}

libmmv::Vec3ui ProblemSideIterator::next()
{
    libmmv::Vec3ui nextCoord = indexToCoordinate(currentVoxelIndex);
    currentVoxelIndex++;
    return nextCoord;
}

bool ProblemSideIterator::hasNext()
{
    return currentVoxelIndex < problem->getNumberOfVoxels();
}

libmmv::Vec3ui ProblemSideIterator::indexToCoordinate(int index)
{
    switch (side) {
    case POSITIVE_X:
        return libmmv::Vec3ui(0, index % problemSize.y, index / problemSize.y);
    case POSITIVE_Y:
        return libmmv::Vec3ui(index % problemSize.x, 0, index / problemSize.x);
    case POSITIVE_Z:
        return libmmv::Vec3ui(index % problemSize.x, index / problemSize.x, 0);
    case NEGATIVE_X:
        return libmmv::Vec3ui(problemSize.x - 1, index % problemSize.y, index / problemSize.y);
    case NEGATIVE_Y:
        return libmmv::Vec3ui(index % problemSize.x, problemSize.y - 1, index / problemSize.x);
    case NEGATIVE_Z:
        return libmmv::Vec3ui(index % problemSize.x, index / problemSize.x, problemSize.z - 1);
    }

    throw std::runtime_error("No valid side provided");
}

// Enumerate the vertices of a particular voxel side based on an index between 0 and 4, 0 being lower-left, 1 being lower-right etc
libmmv::Vec3i ProblemSideIterator::sideIndexToVertexCoordinateOffset(unsigned int sideIndex) {
    switch (side) {
    case POSITIVE_X:
        return offsetsPosX[sideIndex];
    case POSITIVE_Y:
        return offsetsPosY[sideIndex];
    case POSITIVE_Z:
        return offsetsPosZ[sideIndex];
    case NEGATIVE_X:
        return offsetsNegX[sideIndex];
    case NEGATIVE_Y:
        return offsetsNegY[sideIndex];
    case NEGATIVE_Z:
        return offsetsNegZ[sideIndex];
    default:
        throw std::runtime_error("Illegal problem side encountered");
    }
}
