#include <stdafx.h>
#include "ProblemFragment.h"

ProblemFragment::ProblemFragment(ettention::Vec3ui& centerVertexCoord, std::vector<Material*>& materials) :
    m_centerVertexCoord(centerVertexCoord),
    m_materials(materials),
    m_key(&m_materials)
{
}

ProblemFragment::ProblemFragment(ettention::Vec3ui& centerVertexCoord) :
    m_centerVertexCoord(centerVertexCoord),
    m_materials(8, &Material::EMPTY),
    m_key(&m_materials)
{
}

ProblemFragment::~ProblemFragment() {

}

void ProblemFragment::setMaterial(unsigned int index, Material& mat) {
    setMaterial(index, &mat);
}

void ProblemFragment::setMaterial(unsigned int index, Material* mat) {
    m_materials[index] = mat;
    m_key = ProblemFragmentKey(&m_materials);
}

const ProblemFragmentKey& ProblemFragment::key() const {
    return m_key;
}

bool ProblemFragment::containsMixedMaterials() const {
    for (int i = 1; i < 8; i++) {
        if (m_materials[i] != m_materials[0]) {
            return true;
        }
    }
    return false;
}
