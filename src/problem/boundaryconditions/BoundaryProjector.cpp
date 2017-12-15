#include "stdafx.h"
#include "BoundaryProjector.h"
#include "problem/DiscreteProblem.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include <iostream>

// The maximum depth (in voxels) to trace each ray before assuming there is no exposed surface
#define RAYCAST_MAX_DEPTH 20

// The maximum "thickness" of the space where boundary conditions may be applied. Ex. if z==1 is the top-most voxel
// receiving a boundary condition then no voxels beyond z==11 can receive this boundary condition
#define RAYCAST_MAX_DEPTH_FROM_FIRST_HIT 6

BoundaryProjector::BoundaryProjector(DiscreteProblem* p) :
    problem(p),
    problemSize(p->getSize())
{
}

BoundaryProjector::~BoundaryProjector()
{
}

void BoundaryProjector::projectDirichletBoundaryAlongNegZ(DirichletBoundary* condition)
{
    libmmv::Vec3i projectionStep(0, 0, -1);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int x = 0; x < problemSize.x; x++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(x, y, problemSize.z - 1);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels);
        }
    }

    auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
        return a.z < b.z;
    });
    unsigned int topZ = maxElement->z;
    unsigned int numBoundaryVoxels = 0;
    for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
        // We only want to fix the voxels near the outer edge of the space
        if (abs((int)topZ - (int)it->z) < RAYCAST_MAX_DEPTH_FROM_FIRST_HIT) {
            libmmv::Vec3ui vertexCoord(*it);
            vertexCoord.z++; //go to the top layer of vertices for this voxel
            problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
            vertexCoord.x++;
            problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
            vertexCoord.y++;
            problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
            vertexCoord.x--;
            problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
            numBoundaryVoxels++;
        }
    }
}

void BoundaryProjector::projectNeumannStressAlongPosZ(REAL neumannStressPerSquareMeter)
{
    // Project rays from z==0 up along the Z axis, find the first non-null-material voxel for each x,y position to build the exposed surface
    libmmv::Vec3i projectionStep(0, 0, 1);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int x = 0; x < problemSize.x; x++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(x, y, 0);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels);
        }
    }

    // Only consider voxels within 20 layers of the first non-null-material voxel encountered, to ensure we don't add a boundary to surfaces far away from the force origin
    auto minElement = std::min_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
        return a.z > b.z;
    });
    unsigned int zLayerCutoff = minElement->z + RAYCAST_MAX_DEPTH_FROM_FIRST_HIT;
    std::vector<libmmv::Vec3ui> filteredSurface;
    std::copy_if(surfaceVoxels.begin(), surfaceVoxels.end(), std::back_inserter(filteredSurface), [zLayerCutoff](const libmmv::Vec3ui& a) -> bool {
        return a.z < zLayerCutoff;
    });
    
    // Scale the total stress in sqm to stress per vertex, depends on total surface area hit by the ray casting above
    REAL surfaceAreaPerVoxelInSqMeters = asREAL(problem->getVoxelSize().x * problem->getVoxelSize().y);
    REAL totalSurfaceAreaInSqMeters = surfaceAreaPerVoxelInSqMeters * filteredSurface.size();
    REAL neumannStressPerVoxel = neumannStressPerSquareMeter * totalSurfaceAreaInSqMeters;
    REAL neumannStressPerVertex = neumannStressPerVoxel * asREAL(0.25); // exposed surface of each voxel has 4 vertices
    
    // For each relevant voxel, add the stress to each of its 4 vertices lying on the exposed surface
    unsigned int numBoundaryVoxels = 0;
    for (auto it = filteredSurface.begin(); it != filteredSurface.end(); it++) {
        libmmv::Vec3ui vertexCoord(*it);
        NeumannBoundary stress(libmmv::Vec3<REAL>(0, 0, neumannStressPerVertex));
        problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
        vertexCoord.x++;
        problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
        vertexCoord.y++;
        problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
        vertexCoord.x--;
        problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
        numBoundaryVoxels++;
    }
}

// Projects a ray along updateStep direction checking each voxel along the way to find the first non-null-material voxel
void BoundaryProjector::projectRayToFindSurface(libmmv::Vec3ui & origin, const libmmv::Vec3i* updateStep, std::vector<libmmv::Vec3ui>* surfaceCandidates)
{
    libmmv::Vec3ui rayPos(origin);
    unsigned int layersTraversed = 0;
    while (layersTraversed < RAYCAST_MAX_DEPTH && rayPos.x < problemSize.x && rayPos.x >= 0 && rayPos.y < problemSize.y && rayPos.y >= 0 && rayPos.z < problemSize.z && rayPos.z >= 0) {
        Material* voxelMat = problem->getMaterial(rayPos);
        if (voxelMat->id != Material::EMPTY.id) {
            surfaceCandidates->push_back(rayPos);
            return;
        }
        rayPos = rayPos + *updateStep;
        layersTraversed++;
    }
}

