#include <stdafx.h>
#include "DiscreteProblem.h"
#include "problem/ProblemFragment.h"

DiscreteProblem::DiscreteProblem(ettention::Vec3ui size, ettention::Vec3d voxelSize) :
    m_size(size),
    m_voxelSize(voxelSize),
    m_numberOfCells(size.x*size.y*size.z),
    m_materials(m_numberOfCells, &Material::EMPTY)
{

}

DiscreteProblem::~DiscreteProblem() {

}

void DiscreteProblem::setMaterial(ettention::Vec3ui& coordinate, Material& mat) {
    unsigned int index = mapToIndex(coordinate);
    setMaterial(index, mat);
}

void DiscreteProblem::setMaterial(unsigned int index, Material& mat) {
    m_materials[index] = &mat;
}

unsigned int DiscreteProblem::mapToIndex(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        throw std::invalid_argument("given coordinate cannot be mapped to an index because it is outside the problem space");
    }
    return coordinate.x + coordinate.y * m_size.x + coordinate.z * m_size.x * m_size.y;
}

ettention::Vec3ui DiscreteProblem::mapToCoordinate(unsigned int index) const {
    ettention::Vec3ui coord(index % m_size.x, (index / m_size.x) % m_size.y, index / (m_size.x * m_size.y));

    if (outOfBounds(coord)) {
        throw std::invalid_argument("given index could not be mapped to a valid coordinate because it is outside the problem space");
    }
    return coord;
}


Material* DiscreteProblem::getMaterial(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        return &Material::EMPTY;
    }
    unsigned int index = mapToIndex(coordinate);
    return getMaterial(index);
}

Material* DiscreteProblem::getMaterial(unsigned int index) const {
    return m_materials.at(index);
}

ettention::Vec3d DiscreteProblem::getVoxelSize() const {
    return ettention::Vec3d(m_voxelSize);
}

ettention::Vec3ui DiscreteProblem::getSize() const {
    return ettention::Vec3ui(m_size);
}

std::vector<Material*>* DiscreteProblem::getMaterialVector() {
    return &m_materials;
}

inline bool DiscreteProblem::outOfBounds(ettention::Vec3ui& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= m_size.x || coordinate.y < 0 || coordinate.y >= m_size.y || coordinate.z < 0 || coordinate.z >= m_size.z;
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
