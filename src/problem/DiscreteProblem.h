#pragma once
#include <vector>
#include "libmmv/math/Vec3.h"
#include "problem/ProblemFragment.h"

class DiscreteProblem {

public:
    DiscreteProblem(ettention::Vec3ui size, ettention::Vec3d voxelSize);
    ~DiscreteProblem();

    void setMaterial(ettention::Vec3ui& coordinate, Material& mat);
    void setMaterial(unsigned int index, Material& mat);

    Material* getMaterial(ettention::Vec3ui& coordinate) const;
    Material* getMaterial(unsigned int index) const;
    ettention::Vec3d getVoxelSize() const;
    ettention::Vec3ui getSize() const;
    std::vector<Material*>* getMaterialVector();

    unsigned int mapToIndex(ettention::Vec3ui& coordinate) const;
    ettention::Vec3ui mapToCoordinate(unsigned int index) const;

    ProblemFragment extractLocalProblem(ettention::Vec3ui centerCoord) const;

private:
    const ettention::Vec3ui m_size;
    const ettention::Vec3d m_voxelSize;
    const unsigned int m_numberOfCells;
    std::vector<Material*> m_materials;

    inline bool outOfBounds(ettention::Vec3ui& coordinate) const;
};
