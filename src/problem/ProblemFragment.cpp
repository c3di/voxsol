#include <stdafx.h>
#include "ProblemFragment.h"
#include "material/Material.h"
#include "material/MaterialConfiguration.h"

ProblemFragment::ProblemFragment(libmmv::Vec3ui& centerVertexCoord, std::vector<Material*>& mats) :
    centerVertexCoord(centerVertexCoord),
    materials(mats)
{
}

ProblemFragment::ProblemFragment(libmmv::Vec3ui& centerVertexCoord) :
    centerVertexCoord(centerVertexCoord),
    materials(8, &Material::EMPTY)
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
}

void ProblemFragment::setDirichletBoundary(const DirichletBoundary& fixed) {
    dirichletBoundaryCondition = DirichletBoundary(fixed);
}

void ProblemFragment::setNeumannBoundary(const NeumannBoundary& stress) {
    neumannBoundaryCondition = NeumannBoundary(stress);
}

const DirichletBoundary& ProblemFragment::getDirichletBoundaryCondition() const {
    return dirichletBoundaryCondition;
}

const NeumannBoundary& ProblemFragment::getNeumannBoundaryCondition() const {
    return neumannBoundaryCondition;
}

const std::vector<Material*>* ProblemFragment::getMaterials() const {
    return &materials;
}

const MaterialConfiguration ProblemFragment::getMaterialConfiguration() const {
    return MaterialConfiguration(this);
}

libmmv::Vec3ui ProblemFragment::getCenterVertex() const {
    return libmmv::Vec3ui(centerVertexCoord);
}

bool ProblemFragment::containsMixedMaterials() const {
    for (int i = 1; i < 8; i++) {
        if (materials[i] != materials[0]) {
            return true;
        }
    }
    return false;
}
