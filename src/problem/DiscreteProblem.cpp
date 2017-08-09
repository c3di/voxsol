#include <stdafx.h>
#include "DiscreteProblem.h"
#include "problem/ProblemFragment.h"

DiscreteProblem::DiscreteProblem(ettention::Vec3ui size, ettention::Vec3d voxelSize, MaterialDictionary* matDict) :
    size(size),
    voxelSize(voxelSize),
    numberOfCells(size.x*size.y*size.z),
    materialIds(numberOfCells, Material::EMPTY.id),
    materialDictionary(matDict)
{
    
}

DiscreteProblem::~DiscreteProblem() {

}

void DiscreteProblem::setMaterial(ettention::Vec3ui& coordinate, unsigned char matId) {
    unsigned int index = mapToIndex(coordinate);
    setMaterial(index, matId);
}

void DiscreteProblem::setMaterial(unsigned int index, unsigned char matId) {
    materialIds[index] = matId;
}

unsigned int DiscreteProblem::mapToIndex(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        throw std::invalid_argument("given coordinate cannot be mapped to an index because it is outside the problem space");
    }
    return coordinate.x + coordinate.y * size.x + coordinate.z * size.x * size.y;
}

ettention::Vec3ui DiscreteProblem::mapToCoordinate(unsigned int index) const {
    return ettention::Vec3ui(index % size.x, (index / size.x) % size.y, index / (size.x * size.y));
}


Material* DiscreteProblem::getMaterial(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        return &Material::EMPTY;
    }
    unsigned int index = mapToIndex(coordinate);
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
    return ettention::Vec3ui(size);
}

std::vector<unsigned char>* DiscreteProblem::getMaterialIdVector() {
    return &materialIds;
}

bool DiscreteProblem::outOfBounds(ettention::Vec3ui& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= size.x || coordinate.y < 0 || coordinate.y >= size.y || coordinate.z < 0 || coordinate.z >= size.z;
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

    return ProblemFragment(centerCoord, mats);
}
