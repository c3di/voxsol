#include "stdafx.h"
#include "BoundaryProjector.h"
#include "problem/DiscreteProblem.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "problem/boundaryconditions/ProblemSideIterator.h"
#include <iostream>

/**
    Projects boundary conditions onto the problem by tracing rays through the volume from the given direction. 
    The first non-null voxel each ray hits is given the boundary condition, assuming it is within the maximum 
    depth.
*/

BoundaryProjector::BoundaryProjector(DiscreteProblem* p) :
    problem(p),
    problemSize(p->getSize())
{
}

BoundaryProjector::~BoundaryProjector()
{
}

void BoundaryProjector::setMaxProjectionDepth(unsigned int depthFromTop, unsigned int depthFromTopHit)
{
    maxDepthFromTop = depthFromTop;
    maxDepthFromTopmostHit = depthFromTopHit;
}

void BoundaryProjector::projectDirichletBoundary(DirichletBoundary* condition, const ProblemSide side)
{
	std::vector<libmmv::Vec3ui> surfaceVoxels;
	ProblemSideIterator it(problem, side);
	while (it.hasNext()) {
		libmmv::Vec3ui origin = it.next();
		projectRayToFindSurface(origin, side, &surfaceVoxels, 255);
	}

	if (surfaceVoxels.size() > 0) {
		auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
			return a.z > b.z; //
			});
		unsigned int topZ = maxElement->z; //
		unsigned int numBoundaryVoxels = 0;
		for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
			// We only want to fix the voxels near the outer edge of the space
			if (abs((int)topZ - (int)it->z) <= (int)maxDepthFromTopmostHit) { //
				libmmv::Vec3ui vertexCoord(*it);
				vertexCoord.z++; //go to the top layer of vertices for this voxel   //
				problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
				vertexCoord.x++; //
				problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
				vertexCoord.y++; //
				problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
				vertexCoord.x--; //
				problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
				numBoundaryVoxels++;
			}
		}
	}

}

void BoundaryProjector::projectNeumannBoundary(REAL totalForce, const ProblemSide side, unsigned char materialFilter) {

}
/*
void BoundaryProjector::projectDirichletBoundaryAlongNegZ(DirichletBoundary* condition)
{
    libmmv::Vec3i projectionStep(0, 0, -1); // idea: should be passed as parameter?
    std::vector<libmmv::Vec3ui> surfaceVoxels;
	
	// idea: implement an iterator that internally stored the axis aligned bouding biox 
	// it iterates
    for (unsigned int y = 0; y < problemSize.y; y++) { //
        for (unsigned int x = 0; x < problemSize.x; x++) { //
            libmmv::Vec3ui origin(x, y, problemSize.z-1); //
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, 255);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.z > b.z; //
        });
        unsigned int topZ = maxElement->z; //
        unsigned int numBoundaryVoxels = 0;
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            // We only want to fix the voxels near the outer edge of the space
            if (abs((int)topZ - (int)it->z) <= (int)maxDepthFromTopmostHit) { //
                libmmv::Vec3ui vertexCoord(*it);
                vertexCoord.z++; //go to the top layer of vertices for this voxel   //
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.x++; //
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.y++; //
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.x--; //
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                numBoundaryVoxels++;
            }
        }
    }
	std::cout << "Projected Dirichlet boundary onto " << surfaceVoxels.size() << " voxels along -Z\n"; //
}
*/
// Projects a ray along updateStep direction checking each voxel along the way to find the first non-null-material voxel
void BoundaryProjector::projectRayToFindSurface(libmmv::Vec3ui & origin, const ProblemSide side, std::vector<libmmv::Vec3ui>* surfaceCandidates, unsigned char matIdFilter)
{
	libmmv::Vec3i projectionStep;
	switch (side) {
	case POSITIVE_X:
		projectionStep = libmmv::Vec3i(1, 0, 0);
	case POSITIVE_Y:
		projectionStep = libmmv::Vec3i(0, 1, 0);
	case POSITIVE_Z:
		projectionStep = libmmv::Vec3i(0, 0, 1);
	case NEGATIVE_X:
		projectionStep = libmmv::Vec3i(-1, 0, 0);
	case NEGATIVE_Y:
		projectionStep = libmmv::Vec3i(0, -1, 0);
	case NEGATIVE_Z:
		projectionStep = libmmv::Vec3i(0, 0, -1);
	}

    libmmv::Vec3ui rayPos(origin);
    unsigned int layersTraversed = 0;
    while (layersTraversed < maxDepthFromTop && rayPos.x < problemSize.x && rayPos.x >= 0 && rayPos.y < problemSize.y && rayPos.y >= 0 && rayPos.z < problemSize.z && rayPos.z >= 0) {
        Material* voxelMat = problem->getMaterial(rayPos);
        if (voxelMat->id != Material::EMPTY.id) {
            if (matIdFilter == 255 || voxelMat->id == matIdFilter) {
                surfaceCandidates->push_back(rayPos);
            }
            return;
        }
        rayPos = rayPos + projectionStep;
        layersTraversed++;
    }
}

