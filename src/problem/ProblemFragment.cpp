#include <stdafx.h>
#include "ProblemFragment.h"
#include "material/Material.h"

ProblemFragment::ProblemFragment(ettention::Vec3ui& centerVertexCoord, std::vector<Material*>& mats) :
    centerVertexCoord(centerVertexCoord),
    materials(mats),
    materialConfig(&mats)
{
}

ProblemFragment::ProblemFragment(ettention::Vec3ui& centerVertexCoord) :
    centerVertexCoord(centerVertexCoord),
    materials(8, &Material::EMPTY),
    materialConfig(&materials)
{
}

ProblemFragment::~ProblemFragment() {

}

REAL ProblemFragment::mu(unsigned int cell) const {
    return materials[cell]->mu;
}

REAL ProblemFragment::lambda(unsigned int cell) const {
    return materials[cell]->lambda;
}

void ProblemFragment::setMaterial(unsigned int index, Material& mat) {
    setMaterial(index, &mat);
}

void ProblemFragment::setMaterial(unsigned int index, Material* mat) {
    materials[index] = mat;
    updateMaterialConfig();
}

void ProblemFragment::setDirichletBoundary(const DirichletBoundary& fixed) {
    dirichletBoundaryCondition = DirichletBoundary(fixed);
    updateMaterialConfig();
}

const DirichletBoundary& ProblemFragment::getDirichletBoundaryConditions() const {
    return dirichletBoundaryCondition;
}

const MaterialConfiguration& ProblemFragment::getMaterialConfiguration() const {
    return materialConfig;
}

ettention::Vec3ui ProblemFragment::getCenterVertex() const {
    return ettention::Vec3ui(centerVertexCoord);
}

bool ProblemFragment::containsMixedMaterials() const {
    for (int i = 1; i < 8; i++) {
        if (materials[i] != materials[0]) {
            return true;
        }
    }
    return false;
}

void ProblemFragment::updateMaterialConfig() {
    materialConfig = MaterialConfiguration(&materials, dirichletBoundaryCondition);
}
