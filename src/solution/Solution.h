#pragma once
#include <libmmv/math/Vec3.h>
#include "material/MaterialConfigurationEquations.h"
#include <unordered_map>
#include <vector>

class DiscreteProblem;

class Solution {
public:

    Solution(DiscreteProblem& problem);
    ~Solution();

    void computeMaterialConfigurationEquations();

    unsigned int mapToIndex(ettention::Vec3ui& coordinate) const;
    ettention::Vec3ui mapToCoordinate(unsigned int index) const;

    const std::vector<ConfigId>* getMaterialConfigurationEquationIds() const;
    const std::vector<MaterialConfigurationEquations>* getMaterialConfigurationEquations() const;
    std::vector<REAL>* getDisplacements();

protected:
    const ettention::Vec3ui size;
    const ettention::Vec3<REAL> voxelSize;
    const DiscreteProblem* const problem;
    std::vector<ConfigId> matConfigEquationIds;
    std::vector<REAL> displacements;
    std::vector<MaterialConfigurationEquations> matConfigEquations;

    void gatherUniqueMaterialConfigurations();
    void computeEquationsForUniqueMaterialConfigurations();

    bool outOfBounds(ettention::Vec3ui& coordinate) const;
};
