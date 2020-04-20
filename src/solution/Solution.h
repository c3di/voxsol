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
    const libmmv::Vec3<REAL> getVoxelSize() const;

    const std::vector<MaterialConfigurationEquations>* getMaterialConfigurationEquations() const;
    std::vector<Vertex>* getVertices();
    Vertex getVertexAt(VertexCoordinate coord);

    void disableMaterialConfigurationCaching();

    DiscreteProblem* getProblem();

protected:
    bool doCacheMaterialConfigurations = true;
    const libmmv::Vec3ui size;
    const libmmv::Vec3<REAL> voxelSize;
    DiscreteProblem* problem;
    std::vector<Vertex> vertices;
    std::vector<MaterialConfigurationEquations> matConfigEquations;

    struct UniqueConfig {
        ConfigId equationId = 0;
        int numInstancesInProblem = 0;
    };

    void gatherUniqueMaterialConfigurations();
    void computeEquationsForUniqueMaterialConfigurations();
    void applyInitialDisplacements();

    bool outOfBounds(libmmv::Vec3ui& coordinate) const;

    void createVoidMaterialConfiguration(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation);
    void scanSolutionForUniqueConfigurations(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation);
    void sortUniqueConfigurationsByFrequency(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation);
    void assignConfigurationIdsToVertices(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation);
};
