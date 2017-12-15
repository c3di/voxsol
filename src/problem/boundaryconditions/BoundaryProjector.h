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

    //void projectDirichletBoundaryAlongNegX(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongNegX(NeumannBoundary* condition);
    //void projectDirichletBoundaryAlongNegY(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongNegY(NeumannBoundary* condition);
    void projectDirichletBoundaryAlongNegZ(DirichletBoundary* condition);
    void projectNeumannStressAlongPosZ(REAL neumannStressPerSquareMeter);

    //void projectDirichletBoundaryAlongPosX(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongPosX(NeumannBoundary* condition);
    //void projectDirichletBoundaryAlongPosY(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongPosY(NeumannBoundary* condition);
    //void projectDirichletBoundaryAlongPosZ(DirichletBoundary* condition);
    //void projectNeumannBoundaryAlongPosZ(NeumannBoundary* condition);

protected:
    DiscreteProblem* problem;
    libmmv::Vec3ui problemSize;

    void projectRayToFindSurface(libmmv::Vec3ui& origin, const libmmv::Vec3i* updateStep, std::vector<libmmv::Vec3ui>* surfaceCandidates);

    REAL getNeumannStressPerVoxel(REAL neumannStressPerSquareMeter, int numberOfVoxelsHit);
};