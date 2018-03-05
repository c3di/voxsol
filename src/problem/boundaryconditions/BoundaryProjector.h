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

    //void projectDirichletBoundaryAlongNegX(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongNegX(NeumannBoundary* condition);
    //void projectDirichletBoundaryAlongNegY(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongNegY(NeumannBoundary* condition);
    void projectDirichletBoundaryAlongNegZ(DirichletBoundary* condition);
    void projectNeumannStressAlongNegZ(REAL totalNeumannStress);

    void projectDirichletBoundaryAlongPosX(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongPosX(NeumannBoundary* condition);
    //void projectDirichletBoundaryAlongPosY(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongPosY(NeumannBoundary* condition);
    //void projectDirichletBoundaryAlongPosZ(DirichletBoundary* condition);
    void projectNeumannStressAlongPosZ(REAL totalNeumannStress);

protected:
    DiscreteProblem* problem;
    libmmv::Vec3ui problemSize;
    unsigned int maxDepthFromTop = 20;
    unsigned int maxDepthFromTopmostHit = 6;


    void projectRayToFindSurface(libmmv::Vec3ui& origin, const libmmv::Vec3i* updateStep, std::vector<libmmv::Vec3ui>* surfaceCandidates);

    REAL getNeumannStressPerVoxel(REAL neumannStressPerSquareMeter, int numberOfVoxelsHit);
};