#pragma once
#include "libmmv/math/Vec3.h"
#include "material/MaterialConfiguration.h"
#include "problem/DirichletBoundary.h"
#include <vector>

class Material;

class ProblemFragment {
public:

    ProblemFragment(ettention::Vec3ui& centerVertexCoord, std::vector<Material*>& materials);
    ProblemFragment(ettention::Vec3ui& centerVertexCoord);
    ~ProblemFragment();

    void setMaterial(unsigned int index, Material& mat);
    void setMaterial(unsigned int index, Material* mat);
    void setDirichletBoundary(const DirichletBoundary& condition);

    const MaterialConfiguration& getMaterialConfiguration() const;
    bool containsMixedMaterials() const;
    ettention::Vec3ui getCenterVertex() const;
    const DirichletBoundary& getDirichletBoundaryConditions() const;

    REAL mu(unsigned int cell) const;
    REAL lambda(unsigned int cell) const;

private:
    DirichletBoundary dirichletBoundaryCondition;
    const ettention::Vec3ui centerVertexCoord;
    std::vector<Material*> materials;
    MaterialConfiguration materialConfig;

    void updateMaterialConfig();
};
