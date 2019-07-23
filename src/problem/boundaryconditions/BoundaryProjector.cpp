#include "stdafx.h"
#include "BoundaryProjector.h"
#include "problem/DiscreteProblem.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
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

void BoundaryProjector::projectDirichletBoundaryAlongNegZ(DirichletBoundary* condition)
{
    libmmv::Vec3i projectionStep(0, 0, -1); //
    std::vector<libmmv::Vec3ui> surfaceVoxels;
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

void BoundaryProjector::projectDirichletBoundaryAlongPosZ(DirichletBoundary* condition, unsigned char matIdFilter)
{
    libmmv::Vec3i projectionStep(0, 0, 1);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int x = 0; x < problemSize.x; x++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(x, y, 0);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, matIdFilter);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto minElement = std::min_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.z < b.z;
        });
        unsigned int topZ = minElement->z;
        unsigned int numBoundaryVoxels = 0;
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            // We only want to fix the voxels near the outer edge of the space
            if (abs((int)topZ - (int)it->z) <= (int)maxDepthFromTopmostHit) {
                libmmv::Vec3ui vertexCoord(*it);
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
    std::cout << "Projected Dirichlet boundary onto " << surfaceVoxels.size() << " voxels along +Z\n";
}

void BoundaryProjector::projectNeumannStressAlongNegZ(REAL totalNeumannForce, unsigned char matIdFilter)
{
    // Project rays from z==top down along the Z axis, find the first non-null-material voxel for each x,y position to build the exposed surface
    libmmv::Vec3i projectionStep(0, 0, -1); //
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int x = 0; x < problemSize.x; x++) { //
        for (unsigned int y = 0; y < problemSize.y; y++) { //
            libmmv::Vec3ui origin(x, y, problemSize.z-1); //
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, matIdFilter);
        }
    }

    std::cout << "Found " << surfaceVoxels.size() << " voxels for Neumann boundary max depth " << (int)maxDepthFromTop << std::endl;
    if (surfaceVoxels.size() > 0) {
        auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.z > b.z; //
        });
        unsigned int zLayerCutoff = maxElement->z - maxDepthFromTopmostHit; //
        std::vector<libmmv::Vec3ui> filteredSurface;
        std::copy_if(surfaceVoxels.begin(), surfaceVoxels.end(), std::back_inserter(filteredSurface), [zLayerCutoff](const libmmv::Vec3ui& a) -> bool { 
            return a.z > zLayerCutoff; //
        });

        REAL neumannForcePerVoxel = totalNeumannForce / filteredSurface.size();
        REAL neumannForcePerVertex = neumannForcePerVoxel * asREAL(0.25); // exposed surface of each voxel has 4 vertices

        // For each relevant voxel, add the stress to each of its 4 vertices lying on the exposed surface
        unsigned int numBoundaryVoxels = 0;
        for (auto it = filteredSurface.begin(); it != filteredSurface.end(); it++) {
            libmmv::Vec3ui vertexCoord(*it);
            NeumannBoundary stress(libmmv::Vec3<REAL>(0, 0, neumannForcePerVertex)); //
            vertexCoord.z++; //
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.x++; //
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.y++; //
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.x--; //
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            numBoundaryVoxels++;
        }
        std::cout << "Projected Neumann boundary onto " << filteredSurface.size() << " voxels along -Z\n"; //
    }

}

void BoundaryProjector::projectNeumannStressAlongPosZ(REAL totalNeumannForce, unsigned char matIdFilter)
{
    // Project rays from z==0 up along the Z axis, find the first non-null-material voxel for each x,y position to build the exposed surface
    libmmv::Vec3i projectionStep(0, 0, 1);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int x = 0; x < problemSize.x; x++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(x, y, 0);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, matIdFilter);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto minElement = std::min_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.z > b.z;
        });
        unsigned int zLayerCutoff = minElement->z + maxDepthFromTopmostHit;
        std::vector<libmmv::Vec3ui> filteredSurface;
        std::copy_if(surfaceVoxels.begin(), surfaceVoxels.end(), std::back_inserter(filteredSurface), [zLayerCutoff](const libmmv::Vec3ui& a) -> bool {
            return a.z < zLayerCutoff;
        });

        REAL neumannForcePerVoxel = totalNeumannForce / filteredSurface.size();
        REAL neumannForcePerVertex = neumannForcePerVoxel * asREAL(0.25); // exposed surface of each voxel has 4 vertices

        // For each relevant voxel, add the stress to each of its 4 vertices lying on the exposed surface
        unsigned int numBoundaryVoxels = 0;
        for (auto it = filteredSurface.begin(); it != filteredSurface.end(); it++) {
            libmmv::Vec3ui vertexCoord(*it);
            NeumannBoundary stress(libmmv::Vec3<REAL>(0, 0, neumannForcePerVertex));
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.x++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.y++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.x--;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            numBoundaryVoxels++;
        }
        std::cout << "Projected Neumann boundary onto " << filteredSurface.size() << " voxels along +Z\n";
    }

}


void BoundaryProjector::projectDirichletBoundaryAlongPosX(DirichletBoundary * condition)
{
    libmmv::Vec3i projectionStep(1, 0, 0);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(0, y, z);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, 255);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto minElement = std::min_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.x < b.x;
        });
        unsigned int firstX = minElement->x;
        unsigned int numBoundaryVoxels = 0;
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            // We only want to fix the voxels near the outer edge of the space
            if (abs((int)firstX - (int)it->x) <= (int)maxDepthFromTopmostHit) {
                libmmv::Vec3ui vertexCoord(*it);
                //Set boundary conditions for vertices at coord, coord.y+1, coord.z+1 and coord.y+1 & coord.z+1
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.y++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.z++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.y--;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                numBoundaryVoxels++;
            }
        }
    }
    std::cout << "Projected Dirichlet boundary onto " << surfaceVoxels.size() << " voxels along +X\n";
}

void BoundaryProjector::projectNeumannStressAlongPosX(REAL totalNeumannForce, unsigned char matIdFilter)
{
    // Project rays from x==0 up along the X axis, find the first non-null-material voxel for each z,y position to build the exposed surface
    libmmv::Vec3i projectionStep(1, 0, 0);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(0, y, z);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, matIdFilter);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto minElement = std::min_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.x > b.x;
        });
        unsigned int xLayerCutoff = minElement->x + maxDepthFromTopmostHit;
        std::vector<libmmv::Vec3ui> filteredSurface;
        std::copy_if(surfaceVoxels.begin(), surfaceVoxels.end(), std::back_inserter(filteredSurface), [xLayerCutoff](const libmmv::Vec3ui& a) -> bool {
            return a.x < xLayerCutoff;
        });

        REAL neumannForcePerVoxel = totalNeumannForce / filteredSurface.size();
        REAL neumannForcePerVertex = neumannForcePerVoxel * asREAL(0.25); // exposed surface of each voxel has 4 vertices

        // For each relevant voxel, add the stress to each of its 4 vertices lying on the exposed surface
        unsigned int numBoundaryVoxels = 0;
        for (auto it = filteredSurface.begin(); it != filteredSurface.end(); it++) {
            libmmv::Vec3ui vertexCoord(*it);
            NeumannBoundary stress(libmmv::Vec3<REAL>(neumannForcePerVertex, 0, 0));
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.z++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.y++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.z--;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            numBoundaryVoxels++;
        }
        std::cout << "Projected Neumann boundary onto " << filteredSurface.size() << " voxels along +X\n";
    }
}

void BoundaryProjector::projectDirichletBoundaryAlongNegX(DirichletBoundary * condition)
{
    libmmv::Vec3i projectionStep(-1, 0, 0);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(problemSize.x - 1, y, z);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, 255);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.x > b.x;
        });
        unsigned int firstX = maxElement->x;
        unsigned int numBoundaryVoxels = 0;
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            // We only want to fix the voxels near the outer edge of the space
            if (abs((int)firstX - (int)it->x) <= (int)maxDepthFromTopmostHit) {
                libmmv::Vec3ui vertexCoord(*it);
                //Set boundary conditions for vertices at coord, coord.y+1, coord.z+1 and coord.y+1 & coord.z+1
                vertexCoord.x++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.y++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.z++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.y--;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                numBoundaryVoxels++;
            }
        }
    }
    std::cout << "Projected Dirichlet boundary onto " << surfaceVoxels.size() << " voxels along -X\n";
}

void BoundaryProjector::projectNeumannStressAlongNegX(REAL totalNeumannForce, unsigned char matIdFilter)
{
    // Project rays from x==top down along the X axis, find the first non-null-material voxel for each x,z position to build the exposed surface
    libmmv::Vec3i projectionStep(-1, 0, 0);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            libmmv::Vec3ui origin(problemSize.x - 1, y, z);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, matIdFilter);
        }
    }

    std::cout << "Found " << surfaceVoxels.size() << " voxels for Neumann boundary max depth " << (int)maxDepthFromTop << std::endl;
    if (surfaceVoxels.size() > 0) {
        auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.x > b.x;
        });
        unsigned int xLayerCutoff = maxElement->x - maxDepthFromTopmostHit;
        std::vector<libmmv::Vec3ui> filteredSurface;
        std::copy_if(surfaceVoxels.begin(), surfaceVoxels.end(), std::back_inserter(filteredSurface), [xLayerCutoff](const libmmv::Vec3ui& a) -> bool {
            return a.x > xLayerCutoff;
        });

        REAL neumannForcePerVoxel = totalNeumannForce / filteredSurface.size();
        REAL neumannForcePerVertex = neumannForcePerVoxel * asREAL(0.25); // exposed surface of each voxel has 4 vertices

        // For each relevant voxel, add the stress to each of its 4 vertices lying on the exposed surface
        unsigned int numBoundaryVoxels = 0;
        for (auto it = filteredSurface.begin(); it != filteredSurface.end(); it++) {
            libmmv::Vec3ui vertexCoord(*it);
            NeumannBoundary stress(libmmv::Vec3<REAL>(neumannForcePerVertex, 0, 0));
            vertexCoord.x++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.z++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.y++;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            vertexCoord.z--;
            problem->setNeumannBoundaryAtVertex(vertexCoord, stress, true);
            numBoundaryVoxels++;
        }
        std::cout << "Projected Neumann boundary onto " << filteredSurface.size() << " voxels along -X\n";
    }
}

void BoundaryProjector::projectDirichletBoundaryAlongPosY(DirichletBoundary * condition)
{
    libmmv::Vec3i projectionStep(0, 1, 0);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int x = 0; x < problemSize.x; x++) {
            libmmv::Vec3ui origin(x, 0, z);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, 255);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto minElement = std::min_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.y < b.y;
        });
        unsigned int firstY = minElement->y;
        unsigned int numBoundaryVoxels = 0;
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            // We only want to fix the voxels near the outer edge of the space
            if (abs((int)firstY - (int)it->y) <= (int)maxDepthFromTopmostHit) {
                libmmv::Vec3ui vertexCoord(*it);
                //Set boundary conditions for vertices at coord, coord.x+1, coord.z+1 and coord.x-1 & coord.z-1
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.x++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.z++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.x--;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                numBoundaryVoxels++;
            }
        }
    }
    std::cout << "Projected Dirichlet boundary onto " << surfaceVoxels.size() << " voxels along +Y\n";
}

void BoundaryProjector::projectNeumannStressAlongPosY(REAL totalNeumannStress, unsigned char matIdFilter)
{
    throw std::exception("Neumann boundary projection along +Y not implemented yet");
}

void BoundaryProjector::projectDirichletBoundaryAlongNegY(DirichletBoundary * condition)
{
    libmmv::Vec3i projectionStep(0, -1, 0);
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int x = 0; x < problemSize.x; x++) {
            libmmv::Vec3ui origin(x, problemSize.y - 1, z);
            projectRayToFindSurface(origin, &projectionStep, &surfaceVoxels, 255);
        }
    }

    if (surfaceVoxels.size() > 0) {
        auto maxElement = std::max_element(surfaceVoxels.begin(), surfaceVoxels.end(), [](const libmmv::Vec3ui& a, const libmmv::Vec3ui& b) -> bool {
            return a.y > b.y;
        });
        unsigned int firstY = maxElement->y;
        unsigned int numBoundaryVoxels = 0;
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            // We only want to fix the voxels near the outer edge of the space
            if (abs((int)firstY - (int)it->y) <= (int)maxDepthFromTopmostHit) {
                libmmv::Vec3ui vertexCoord(*it);
                //Set boundary conditions for vertices at coord, coord.x+1, coord.z+1 and coord.x-1 & coord.z-1
                vertexCoord.y++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.x++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.z++;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                vertexCoord.x--;
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
                numBoundaryVoxels++;
            }
        }
    }
    std::cout << "Projected Dirichlet boundary onto " << surfaceVoxels.size() << " voxels along -Y\n";
}

void BoundaryProjector::projectNeumannStressAlongNegY(REAL totalNeumannStress, unsigned char matIdFilter)
{
    throw std::exception("Neumann boundary projection along -Y not implemented yet");
}

// Projects a ray along updateStep direction checking each voxel along the way to find the first non-null-material voxel
void BoundaryProjector::projectRayToFindSurface(libmmv::Vec3ui & origin, const libmmv::Vec3i* updateStep, std::vector<libmmv::Vec3ui>* surfaceCandidates, unsigned char matIdFilter)
{
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
        rayPos = rayPos + *updateStep;
        layersTraversed++;
    }
}

