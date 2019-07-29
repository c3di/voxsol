#include "stdafx.h"

#include "ProblemSideIterator.h"
#include "problem/DiscreteProblem.h"

ProblemSideIterator::ProblemSideIterator(DiscreteProblem * problem, ProblemSide side)
	: problem(problem), side(side), problemSize(problem->getSize())
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
		return libmmv::Vec3ui(0, index % problemSize.y, index / problemSize.z);
	case POSITIVE_Y:
		return libmmv::Vec3ui(index % problemSize.x, 0, index / problemSize.z);
	case POSITIVE_Z:
		return libmmv::Vec3ui(index % problemSize.x, index / problemSize.y, 0);
	case NEGATIVE_X:
		return libmmv::Vec3ui(problemSize.x - 1, index % problemSize.y, index / problemSize.z);
	case NEGATIVE_Y:
		return libmmv::Vec3ui(index % problemSize.x, problemSize.y - 1, index / problemSize.z);
	case NEGATIVE_Z:
		return libmmv::Vec3ui(index % problemSize.x, index / problemSize.y, problemSize.z - 1);
	}

	throw std::runtime_error("No valid side provided");
}
