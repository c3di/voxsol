#include <stdafx.h>
#include "ProblemFragment.h"

ProblemFragment::ProblemFragment(ettention::Vec3ui& centerVertexCoord, std::vector<Material*>& mats) :
    centerVertexCoord(centerVertexCoord),
    materials(mats),
    key(&mats)
{
}

ProblemFragment::ProblemFragment(ettention::Vec3ui& centerVertexCoord) :
    centerVertexCoord(centerVertexCoord),
    materials(8, &Material::EMPTY),
    key(&materials)
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
    key = ProblemFragmentKey(&materials);
}

const ProblemFragmentKey& ProblemFragment::getKey() const {
    return key;
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
