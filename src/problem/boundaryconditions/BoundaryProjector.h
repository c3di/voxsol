#pragma once

#include "libmmv/math/Vec3.h"
#include "problem/DiscreteProblem.h"
#include "problem/boundaryconditions/ProblemSideIterator.h"

class DiscreteProblem;
class DirichletBoundary;
class NeumannBoundary;

/**
* Uses raycasting to project a boundary condition onto a surface given an origin and direction. 
* All non-null-material voxels exposed to these rays will be given the boundary condition.
*/
class BoundaryProjector {
public:
    BoundaryProjector(DiscreteProblem* problem, ProblemSide side);
    ~BoundaryProjector();

    // maxAbsoluteDepth: how far rays are traced into the volume to look for a non-empty hit
    void setMaxProjectionDepth(unsigned int maxAbsoluteDepth);
    void setProjectionDirection(ProblemSide side);

	void projectDirichletBoundary(DirichletBoundary* condition);
	void projectNeumannBoundary(REAL totalForce, unsigned char materialFilter = 255);

protected:
    ProblemSide projectFromSide;
    DiscreteProblem* problem;
    libmmv::Vec3ui problemSize;
    unsigned int maxAbsoluteDepth = 20;

    void projectRayToFindSurface(libmmv::Vec3ui& origin, std::vector<libmmv::Vec3ui>* surfaceCandidates, unsigned char matIdFilter);
    
    libmmv::Vec3i getProjectionStepVector();
    NeumannBoundary getNeumannBoundary(REAL forcePerVertex);
    std::string getProjectionDirectionAsString();
};