#include <stdafx.h>
#include "DiscreteProblem.h"
#include "material/MaterialDictionary.h"

DiscreteProblem::DiscreteProblem(libmmv::Vec3ui size, libmmv::Vec3<REAL> voxelSize, MaterialDictionary* matDict) :
    problemSize(size),
    solutionSize(size + libmmv::Vec3ui(1,1,1)),
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

void DiscreteProblem::setNeumannBoundaryAtVertex(VertexCoordinate& coordinate, NeumannBoundary& condition, bool combineIfAlreadyExists) {
    unsigned int index = mapToVertexIndex(coordinate);
    setNeumannBoundaryAtVertex(index, condition, combineIfAlreadyExists);
}

void DiscreteProblem::setNeumannBoundaryAtVertex(unsigned int index, NeumannBoundary& condition, bool combineIfAlreadyExists) {
    if (combineIfAlreadyExists && neumannBoundaryConditions.count(index) > 0) {
        neumannBoundaryConditions[index].combine(condition);
    }
    else {
        neumannBoundaryConditions[index] = condition;
    }
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
    return libmmv::Vec3ui(index % problemSize.x, (index / problemSize.x) % problemSize.y, index / (problemSize.x * problemSize.y));
}

VertexCoordinate DiscreteProblem::mapToVertexCoordinate(unsigned int index) const {
    return libmmv::Vec3ui(index % solutionSize.x, (index / solutionSize.x) % solutionSize.y, index / (solutionSize.x * solutionSize.y));
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

libmmv::Vec3d DiscreteProblem::getVoxelSize() const {
    return libmmv::Vec3d(voxelSize);
}

libmmv::Vec3ui DiscreteProblem::getSize() const {
    return libmmv::Vec3ui(problemSize);
}

unsigned int DiscreteProblem::getNumberOfVoxels() const {
    return problemSize.x * problemSize.y * problemSize.z;
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

std::unordered_map<unsigned int, NeumannBoundary> DiscreteProblem::getNeumannBoundaryMap()
{
    return neumannBoundaryConditions;
}

bool DiscreteProblem::outOfVoxelBounds(VoxelCoordinate& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= problemSize.x || coordinate.y < 0 || coordinate.y >= problemSize.y || coordinate.z < 0 || coordinate.z >= problemSize.z;
}

bool DiscreteProblem::outOfVertexBounds(VertexCoordinate& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= solutionSize.x || coordinate.y < 0 || coordinate.y >= solutionSize.y || coordinate.z < 0 || coordinate.z >= solutionSize.z;
}

ProblemFragment DiscreteProblem::extractLocalProblem(VertexCoordinate& centerCoord) const {
    std::vector<Material*> mats(8);
    
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                VertexCoordinate globalCoordinate(centerCoord.x + x - 1, centerCoord.y + y - 1, centerCoord.z + z -1);
                int matIndex = x + y * 2 + z * 4;
                mats[matIndex] = getMaterial(globalCoordinate);
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

libmmv::Vec3<REAL> DiscreteProblem::getVertexPosition(unsigned int index) const {
    VertexCoordinate coordinate = mapToVertexCoordinate(index);
    return getVertexPosition(coordinate);
}

libmmv::Vec3<REAL> DiscreteProblem::getVertexPosition(VertexCoordinate& coordinates) const {
    REAL x = static_cast<REAL>(coordinates.x) * voxelSize.x;
    REAL y = static_cast<REAL>(coordinates.y) * voxelSize.y;
    REAL z = static_cast<REAL>(coordinates.z) * voxelSize.z;
    return libmmv::Vec3<REAL>(x, y, z);
}
