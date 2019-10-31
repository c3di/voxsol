#include "stdafx.h"
#include "BoundaryProjector.h"
#include "problem/DiscreteProblem.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "problem/boundaryconditions/DisplacementBoundary.h"
#include "problem/boundaryconditions/ProblemSideIterator.h"
#include <iostream>

/**
    Projects boundary conditions onto the problem by tracing rays through the volume from the given direction. 
    The first non-null voxel each ray hits is given the boundary condition, assuming it is within the maximum 
    depth.
*/
BoundaryProjector::BoundaryProjector(DiscreteProblem* p, ProblemSide side) :
    problem(p),
    problemSize(p->getSize()),
    projectFromSide(side)
{
}

BoundaryProjector::~BoundaryProjector()
{
}

void BoundaryProjector::setMaxProjectionDepth(unsigned int maxAbsoluteDepth)
{
    this->maxAbsoluteDepth = maxAbsoluteDepth;
}

void BoundaryProjector::setProjectionDirection(ProblemSide side) {
    projectFromSide = side;
}

void BoundaryProjector::projectDirichletBoundary(DirichletBoundary* condition)
{
	std::vector<libmmv::Vec3ui> surfaceVoxels;
	ProblemSideIterator sideIterator(problem, projectFromSide);
	while (sideIterator.hasNext()) {
		libmmv::Vec3ui origin = sideIterator.next();
		projectRayToFindSurface(origin, &surfaceVoxels, 255);
	}
    unsigned int numBoundaryVoxels = 0;
	if (surfaceVoxels.size() > 0) {
		for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            for (int sideIndex = 0; sideIndex < 4; sideIndex++) {
                libmmv::Vec3ui vertexCoord(*it);
                vertexCoord = vertexCoord + sideIterator.sideIndexToVertexCoordinateOffset(sideIndex);
                problem->setDirichletBoundaryAtVertex(vertexCoord, *condition);
            }
            numBoundaryVoxels++;
		}
	}
    std::cout << "Projected Dirichlet boundary onto " << numBoundaryVoxels << " voxels in " << getProjectionDirectionAsString() << " direction\n";
}

void BoundaryProjector::projectNeumannBoundary(REAL totalForce, unsigned char materialFilter) {
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    ProblemSideIterator sideIterator(problem, projectFromSide);
    while (sideIterator.hasNext()) {
        libmmv::Vec3ui origin = sideIterator.next();
        projectRayToFindSurface(origin, &surfaceVoxels, materialFilter);
    }

    REAL neumannForcePerVoxel = totalForce / surfaceVoxels.size();
    REAL neumannForcePerVertex = neumannForcePerVoxel * asREAL(0.25); // exposed surface of each voxel has 4 vertices

    unsigned int numBoundaryVoxels = 0;
    if (surfaceVoxels.size() > 0) {
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            NeumannBoundary neumannBoundary = getNeumannBoundary(neumannForcePerVertex);
            for (int sideIndex = 0; sideIndex < 4; sideIndex++) {
                libmmv::Vec3ui vertexCoord(*it);
                vertexCoord = vertexCoord + sideIterator.sideIndexToVertexCoordinateOffset(sideIndex);
                problem->setNeumannBoundaryAtVertex(vertexCoord, neumannBoundary, true);
            }
            numBoundaryVoxels++;
        }
    }
    std::cout << "Projected Neumann boundary onto " << numBoundaryVoxels << " voxels in " << getProjectionDirectionAsString() << " direction\n";
}

void BoundaryProjector::projectDisplacementBoundary(DisplacementBoundary* condition, unsigned char materialFilter) {
    std::vector<libmmv::Vec3ui> surfaceVoxels;
    ProblemSideIterator sideIterator(problem, projectFromSide);
    while (sideIterator.hasNext()) {
        libmmv::Vec3ui origin = sideIterator.next();
        projectRayToFindSurface(origin, &surfaceVoxels, materialFilter);
    }

    libmmv::Vec3<REAL> displacement = condition->displacement;

    unsigned int numBoundaryVoxels = 0;
    if (surfaceVoxels.size() > 0) {
        for (auto it = surfaceVoxels.begin(); it != surfaceVoxels.end(); it++) {
            DisplacementBoundary dispBoundary(displacement);
            for (int sideIndex = 0; sideIndex < 4; sideIndex++) {
                libmmv::Vec3ui vertexCoord(*it);
                vertexCoord = vertexCoord + sideIterator.sideIndexToVertexCoordinateOffset(sideIndex);
                problem->setDisplacementBoundaryAtVertex(vertexCoord, dispBoundary);
            }
            numBoundaryVoxels++;
        }
    }
    std::cout << "Projected Displacement boundary onto " << numBoundaryVoxels << " voxels in " << getProjectionDirectionAsString() << " direction\n";
}

libmmv::Vec3<REAL> BoundaryProjector::getDisplacementFromPercent(REAL percentOfDimension) {
    percentOfDimension /= asREAL(100);
    libmmv::Vec3<REAL> displacement(0, 0, 0);

    switch (projectFromSide) {
    case POSITIVE_X:
        displacement.x = asREAL(problem->getSize().x * problem->getVoxelSize().x * percentOfDimension * -1);
        return displacement;
    case NEGATIVE_X:
        displacement.x = asREAL(problem->getSize().x * problem->getVoxelSize().x * percentOfDimension);
        return displacement;
    case POSITIVE_Y:
        displacement.y = asREAL(problem->getSize().y * problem->getVoxelSize().y * percentOfDimension * -1);
        return displacement;
    case NEGATIVE_Y:
        displacement.y = asREAL(problem->getSize().y * problem->getVoxelSize().y * percentOfDimension);
        return displacement;
    case POSITIVE_Z:
        displacement.z = asREAL(problem->getSize().z * problem->getVoxelSize().z * percentOfDimension * -1);
        return displacement;
    case NEGATIVE_Z:
        displacement.z = asREAL(problem->getSize().z * problem->getVoxelSize().z * percentOfDimension);
        return displacement;
    default:
        throw std::runtime_error("Illegal projection direction encountered");
    }
}

std::string BoundaryProjector::getProjectionDirectionAsString() {
    switch (projectFromSide) {
    case POSITIVE_X:
        return "+X";
    case POSITIVE_Y:
        return "+Y";
    case POSITIVE_Z:
        return "+Z";
    case NEGATIVE_X:
        return "-X";
    case NEGATIVE_Y:
        return "-Y";
    case NEGATIVE_Z:
        return "-Z";
    default:
        throw std::runtime_error("Illegal projection direction encountered");
    }
}

NeumannBoundary BoundaryProjector::getNeumannBoundary(REAL forcePerVertex) {
    libmmv::Vec3<REAL> forceVector(0,0,0);

    switch (projectFromSide) {
    case POSITIVE_X:
    case NEGATIVE_X:
        forceVector.x = forcePerVertex;
        break;
    case POSITIVE_Y:
    case NEGATIVE_Y:
        forceVector.y = forcePerVertex;
        break;
    case POSITIVE_Z:
    case NEGATIVE_Z:
        forceVector.z = forcePerVertex;
        break;
    }
    return NeumannBoundary(forceVector);
}

libmmv::Vec3i BoundaryProjector::getProjectionStepVector() {
    switch (projectFromSide) {
    case POSITIVE_X:
        return libmmv::Vec3i(1, 0, 0);
    case POSITIVE_Y:
        return libmmv::Vec3i(0, 1, 0);
    case POSITIVE_Z:
        return libmmv::Vec3i(0, 0, 1);
    case NEGATIVE_X:
        return libmmv::Vec3i(-1, 0, 0);
    case NEGATIVE_Y:
        return libmmv::Vec3i(0, -1, 0);
    case NEGATIVE_Z:
        return libmmv::Vec3i(0, 0, -1);
    default:
        throw std::runtime_error("Illegal projection direction encountered");
    }
}

// Projects a ray along updateStep direction checking each voxel along the way to find the first non-null-material voxel
void BoundaryProjector::projectRayToFindSurface(libmmv::Vec3ui & origin, std::vector<libmmv::Vec3ui>* surfaceCandidates, unsigned char matIdFilter)
{
    libmmv::Vec3i projectionStep = getProjectionStepVector();
    libmmv::Vec3ui rayPos(origin);
    unsigned int layersTraversed = 0;
    while (layersTraversed < maxAbsoluteDepth && rayPos.x < problemSize.x && rayPos.x >= 0 && rayPos.y < problemSize.y && rayPos.y >= 0 && rayPos.z < problemSize.z && rayPos.z >= 0) {
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

