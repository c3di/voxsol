#pragma once
#include "libmmv/math/Vec3.h"
#include "problem/DirichletBoundary.h"
#include "problem/NeumannBoundary.h"
#include <vector>

class Material;
struct MaterialConfiguration;

class ProblemFragment {
public:

    ProblemFragment(ettention::Vec3ui& centerVertexCoord, std::vector<Material*>& materials);
    ProblemFragment(ettention::Vec3ui& centerVertexCoord);
    ~ProblemFragment();

    void setMaterial(unsigned int index, Material& mat);
    void setMaterial(unsigned int index, Material* mat);
    void setDirichletBoundary(const DirichletBoundary& condition);
    void setNeumannBoundary(const NeumannBoundary& condition);

    const std::vector<Material*>* getMaterials() const;
    const MaterialConfiguration getMaterialConfiguration() const;
    bool containsMixedMaterials() const;
    ettention::Vec3ui getCenterVertex() const;
    const DirichletBoundary& getDirichletBoundaryCondition() const;
    const NeumannBoundary& getNeumannBoundaryCondition() const;

    REAL mu(unsigned int cell) const;
    REAL lambda(unsigned int cell) const;

private:
    DirichletBoundary dirichletBoundaryCondition;
    NeumannBoundary neumannBoundaryCondition;
    const ettention::Vec3ui centerVertexCoord;
    std::vector<Material*> materials;
};
