#include <stdafx.h>
#include "ProblemFragment.h"

ProblemFragment::ProblemFragment(ettention::Vec3ui centerVertexCoord) :
    m_centerVertexCoord(centerVertexCoord),
    m_materials(8, &Material::EMPTY)
{
}

ProblemFragment::~ProblemFragment() {

}

void ProblemFragment::setMaterial(unsigned int index, Material& mat) {
    setMaterial(index, &mat);
}

void ProblemFragment::setMaterial(unsigned int index, Material* mat) {
    m_materials[index] = mat;
}

std::string ProblemFragment::getMaterialConfiguration() const {
    std::string configIdentifier;
    for (int i = 0; i < 8; i++) {
        char c = m_materials[i]->m_id;
        configIdentifier += c;
    }
    return configIdentifier;
}

bool ProblemFragment::containsMixedMaterials() const {
    for (int i = 1; i < 8; i++) {
        if (m_materials[i] != m_materials[0]) {
            return true;
        }
    }
    return false;
}
