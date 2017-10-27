#pragma once
#include <libmmv/math/Vec3.h>
#include "material/MaterialConfigurationEquations.h"
#include "Vertex.h"
#include <unordered_map>
#include <vector>
#include <problem\DiscreteProblem.h>


class Solution {
public:

    Solution(DiscreteProblem& problem);
    ~Solution();

    void computeMaterialConfigurationEquations();

    unsigned int mapToIndex(ettention::Vec3ui& coordinate) const;
    ettention::Vec3ui mapToCoordinate(unsigned int index) const;
    const ettention::Vec3ui getSize() const;

    const std::vector<MaterialConfigurationEquations>* getMaterialConfigurationEquations() const;
    std::vector<Vertex>* getVertices();
    std::vector<Vertex>* getDifferences();
    void updateDisplacements(Vertex* serializedVertices);

    DiscreteProblem* getProblem();

protected:
    const ettention::Vec3ui size;
    const ettention::Vec3<REAL> voxelSize;
    DiscreteProblem* problem;
    std::vector<Vertex> vertices;
    std::vector<Vertex> vertexDiff;
    std::vector<MaterialConfigurationEquations> matConfigEquations;

    void gatherUniqueMaterialConfigurations();
    void computeEquationsForUniqueMaterialConfigurations();

    bool outOfBounds(ettention::Vec3ui& coordinate) const;
};
