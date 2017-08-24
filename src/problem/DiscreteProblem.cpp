#include <stdafx.h>
#include "DiscreteProblem.h"
#include "material/MaterialDictionary.h"

DiscreteProblem::DiscreteProblem(ettention::Vec3ui size, ettention::Vec3d voxelSize, MaterialDictionary* matDict) :
    problemSize(size),
    solutionSize(size + ettention::Vec3ui(1,1,1)),
    voxelSize(voxelSize),
    numberOfCells(size.x*size.y*size.z),
    materialIds(numberOfCells, Material::EMPTY.id),
    materialDictionary(matDict)
{
    
}

DiscreteProblem::~DiscreteProblem() {

}

void DiscreteProblem::setMaterial(VoxelCoordinate& coordinate, unsigned char matId) {
    unsigned int index = mapToVoxelIndex(coordinate);
    setMaterial(index, matId);
}

void DiscreteProblem::setMaterial(unsigned int index, unsigned char matId) {
    materialIds[index] = matId;
}

void DiscreteProblem::setDirichletBoundaryAtVertex(VertexCoordinate& coordinate, DirichletBoundary& condition) {
    unsigned int index = mapToVertexIndex(coordinate);
    setDirichletBoundaryAtVertex(index, condition);
}

void DiscreteProblem::setDirichletBoundaryAtVertex(unsigned int index, DirichletBoundary& condition) {
    dirichletBoundaryConditions[index] = condition;
}

void DiscreteProblem::setNeumannBoundaryAtVertex(VertexCoordinate& coordinate, NeumannBoundary& condition) {
    unsigned int index = mapToVertexIndex(coordinate);
    setNeumannBoundaryAtVertex(index, condition);
}

void DiscreteProblem::setNeumannBoundaryAtVertex(unsigned int index, NeumannBoundary& condition) {
    neumannBoundaryConditions[index] = condition;
}

unsigned int DiscreteProblem::mapToVoxelIndex(VoxelCoordinate& coordinate) const {
    if (outOfVoxelBounds(coordinate)) {
        throw std::invalid_argument("coordinate " + coordinate.to_string() + " cannot be mapped to an index because it is outside the voxel space");
    }
    return coordinate.x + coordinate.y * problemSize.x + coordinate.z * problemSize.x * problemSize.y;
}

unsigned int DiscreteProblem::mapToVertexIndex(VertexCoordinate& coordinate) const {
    if (outOfVertexBounds(coordinate)) {
        throw std::invalid_argument("coordinate "+coordinate.to_string()+" cannot be mapped to an index because it is outside the vertex space");
    }
    return coordinate.x + coordinate.y * solutionSize.x + coordinate.z * solutionSize.x * solutionSize.y;
}

VoxelCoordinate DiscreteProblem::mapToVoxelCoordinate(unsigned int index) const {
    return ettention::Vec3ui(index % problemSize.x, (index / problemSize.x) % problemSize.y, index / (problemSize.x * problemSize.y));
}

VertexCoordinate DiscreteProblem::mapToVertexCoordinate(unsigned int index) const {
    return ettention::Vec3ui(index % solutionSize.x, (index / solutionSize.x) % solutionSize.y, index / (solutionSize.x * solutionSize.y));
}

Material* DiscreteProblem::getMaterial(VoxelCoordinate& coordinate) const {
    if (outOfVoxelBounds(coordinate)) {
        return &Material::EMPTY;
    }
    unsigned int index = mapToVoxelIndex(coordinate);
    return getMaterial(index);
}

Material* DiscreteProblem::getMaterial(unsigned int index) const {
    unsigned char matId = materialIds.at(index);
    assert(materialDictionary->contains(matId));
    return materialDictionary->getMaterialById(matId);
}

ettention::Vec3d DiscreteProblem::getVoxelSize() const {
    return ettention::Vec3d(voxelSize);
}

ettention::Vec3ui DiscreteProblem::getSize() const {
    return ettention::Vec3ui(problemSize);
}

std::vector<unsigned char>* DiscreteProblem::getMaterialIdVector() {
    return &materialIds;
}

DirichletBoundary DiscreteProblem::getDirichletBoundaryAtVertex(VertexCoordinate& coordinate) {
    unsigned int index = mapToVertexIndex(coordinate);
    return getDirichletBoundaryAtVertex(index);
}

DirichletBoundary DiscreteProblem::getDirichletBoundaryAtVertex(unsigned int index) {
    if (dirichletBoundaryConditions.count(index) > 0) {
        return dirichletBoundaryConditions[index];
    }
    else {
        return DirichletBoundary(DirichletBoundary::NONE);
    }
}

NeumannBoundary DiscreteProblem::getNeumannBoundaryAtVertex(VertexCoordinate& coordinate) {
    unsigned int index = mapToVertexIndex(coordinate);
    return getNeumannBoundaryAtVertex(index);
}

NeumannBoundary DiscreteProblem::getNeumannBoundaryAtVertex(unsigned int index) {
    if (neumannBoundaryConditions.count(index) > 0) {
        return neumannBoundaryConditions[index];
    }
    else {
        return NeumannBoundary();
    }
}

bool DiscreteProblem::outOfVoxelBounds(VoxelCoordinate& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= problemSize.x || coordinate.y < 0 || coordinate.y >= problemSize.y || coordinate.z < 0 || coordinate.z >= problemSize.z;
}

bool DiscreteProblem::outOfVertexBounds(VertexCoordinate& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= solutionSize.x || coordinate.y < 0 || coordinate.y >= solutionSize.y || coordinate.z < 0 || coordinate.z >= solutionSize.z;
}

ProblemFragment DiscreteProblem::extractLocalProblem(ettention::Vec3ui centerCoord) const {
    std::vector<Material*> mats;
    
    for (int z = -1; z < 1; z++) {
        for (int y = -1; y < 1; y++) {
            for (int x = -1; x < 1; x++) {
                ettention::Vec3ui offset(centerCoord.x + x, centerCoord.y + y, centerCoord.z + z);
                mats.push_back(getMaterial(offset));
            }
        }
    }
    ProblemFragment fragment(centerCoord, mats);
    considerDirichletBoundaryAtLocalProblem(fragment);
    considerNeumannBoundaryAtLocalProblem(fragment);
    return fragment;
}

void DiscreteProblem::considerDirichletBoundaryAtLocalProblem(ProblemFragment& fragment) const {
    unsigned int index = mapToVertexIndex(fragment.getCenterVertex());
    if (dirichletBoundaryConditions.count(index) > 0) {
        fragment.setDirichletBoundary(dirichletBoundaryConditions.at(index));
    }
}

void DiscreteProblem::considerNeumannBoundaryAtLocalProblem(ProblemFragment& fragment) const {
    unsigned int index = mapToVertexIndex(fragment.getCenterVertex());
    if (neumannBoundaryConditions.count(index) > 0) {
        fragment.setNeumannBoundary(neumannBoundaryConditions.at(index));
    }
}
