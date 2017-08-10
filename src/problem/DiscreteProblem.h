#pragma once
#include "libmmv/math/Vec3.h"
#include "problem/ProblemFragment.h"
#include <vector>

class MaterialDictionary;

class DiscreteProblem {

public:
    DiscreteProblem(ettention::Vec3ui size, ettention::Vec3d voxelSize, MaterialDictionary* matDictionary);
    ~DiscreteProblem();

    void setMaterial(ettention::Vec3ui& coordinate, unsigned char matId);
    void setMaterial(unsigned int index, unsigned char matId);

    Material* getMaterial(ettention::Vec3ui& coordinate) const;
    Material* getMaterial(unsigned int index) const;
    ettention::Vec3d getVoxelSize() const;
    ettention::Vec3ui getSize() const;
    std::vector<unsigned char>* getMaterialIdVector();

    unsigned int mapToIndex(ettention::Vec3ui& coordinate) const;
    ettention::Vec3ui mapToCoordinate(unsigned int index) const;
    ProblemFragment extractLocalProblem(ettention::Vec3ui centerCoord) const;

private:
    const ettention::Vec3ui size;
    const ettention::Vec3d voxelSize;
    const unsigned int numberOfCells;
    std::vector<unsigned char> materialIds;
    MaterialDictionary* materialDictionary;

    bool outOfBounds(ettention::Vec3ui& coordinate) const;
};
