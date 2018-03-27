#pragma once
#include <libmmv/math/Vec3.h>
#include "material/MaterialConfigurationEquations.h"
#include "Vertex.h"
#include <unordered_map>
#include <vector>
#include <problem\DiscreteProblem.h>


class Solution {
public:

    Solution(DiscreteProblem* problem);
    ~Solution();

    void computeMaterialConfigurationEquations();

    unsigned int mapToIndex(libmmv::Vec3ui& coordinate) const;
    libmmv::Vec3ui mapToCoordinate(unsigned int index) const;
    const libmmv::Vec3ui getSize() const;

    const std::vector<MaterialConfigurationEquations>* getMaterialConfigurationEquations() const;
    std::vector<Vertex>* getVertices();

    DiscreteProblem* getProblem();

protected:
    const libmmv::Vec3ui size;
    const libmmv::Vec3<REAL> voxelSize;
    DiscreteProblem* problem;
    std::vector<Vertex> vertices;
    std::vector<MaterialConfigurationEquations> matConfigEquations;

    void gatherUniqueMaterialConfigurations();
    void computeEquationsForUniqueMaterialConfigurations();

    bool outOfBounds(libmmv::Vec3ui& coordinate) const;
};
