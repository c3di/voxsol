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
    BoundaryProjector(DiscreteProblem* problem);
    ~BoundaryProjector();

    // depthFromTop determines how far rays are traced to look for a non-empty hit, 
	// depthFromTopHit determines how thick the volume is where hits can apply the 
	// boundary starting from the outer most hit. 
    void setMaxProjectionDepth(unsigned int depthFromTop, unsigned int depthFromFirstHit);

	void projectDirichletBoundary(DirichletBoundary* condition, const ProblemSide side);

	void projectNeumannBoundary(REAL totalForce, const ProblemSide side, unsigned char materialFilter = 255);

    


protected:
    DiscreteProblem* problem;
    libmmv::Vec3ui problemSize;
    unsigned int maxDepthFromTop = 20;
    unsigned int maxDepthFromTopmostHit = 20;


    void projectRayToFindSurface(libmmv::Vec3ui& origin, const ProblemSide side, std::vector<libmmv::Vec3ui>* surfaceCandidates, unsigned char matIdFilter);

    REAL getNeumannStressPerVoxel(REAL neumannStressPerSquareMeter, int numberOfVoxelsHit);
};