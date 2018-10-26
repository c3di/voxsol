#pragma once
#include "libmmv/math/Vec3.h"
#include "problem/DiscreteProblem.h"

class DiscreteProblem;
class DirichletBoundary;
class NeumannBoundary;

/**
* Uses raycasting to project a boundary condition onto a surface given an origin and direction. All non-null-material
* voxels exposed to these rays will be given the boundary condition.
*/
class BoundaryProjector {
public:

    BoundaryProjector(DiscreteProblem* problem);
    ~BoundaryProjector();

    // depthFromTop determines how far rays are traced to look for a non-empty hit, depthFromTopHit determines how thick the volume is 
    // where hits can apply the boundary starting from the outer most hit. 
    void setMaxProjectionDepth(unsigned int depthFromTop, unsigned int depthFromFirstHit);

    void projectDirichletBoundaryAlongNegX(DirichletBoundary* condition);
    void projectNeumannStressAlongNegX(REAL totalNeumannStress, unsigned char matIdFilter = 255);
    void projectDirichletBoundaryAlongNegY(DirichletBoundary* condition);
    void projectNeumannStressAlongNegY(REAL totalNeumannStress, unsigned char matIdFilter = 255);
    void projectDirichletBoundaryAlongNegZ(DirichletBoundary* condition);
    void projectNeumannStressAlongNegZ(REAL totalNeumannStress, unsigned char matIdFilter = 255);

    void projectDirichletBoundaryAlongPosX(DirichletBoundary* condition);
    void projectNeumannStressAlongPosX(REAL totalNeumannStress, unsigned char matIdFilter = 255);
    void projectDirichletBoundaryAlongPosY(DirichletBoundary* condition);
    void projectNeumannStressAlongPosY(REAL totalNeumannStress, unsigned char matIdFilter = 255);
    void projectDirichletBoundaryAlongPosZ(DirichletBoundary* condition, unsigned char matIdFilter = 255);
    void projectNeumannStressAlongPosZ(REAL totalNeumannStress, unsigned char matIdFilter = 255);

protected:
    DiscreteProblem* problem;
    libmmv::Vec3ui problemSize;
    unsigned int maxDepthFromTop = 20;
    unsigned int maxDepthFromTopmostHit = 20;


    void projectRayToFindSurface(libmmv::Vec3ui& origin, const libmmv::Vec3i* updateStep, std::vector<libmmv::Vec3ui>* surfaceCandidates, unsigned char matIdFilter);

    REAL getNeumannStressPerVoxel(REAL neumannStressPerSquareMeter, int numberOfVoxelsHit);
};