#pragma once
#include <vector>
#include "libmmv/math/Vec3.h"
#include "problem/Material.h"
#include "ProblemFragmentKey.h"

class ProblemFragment {
public:

    ProblemFragment(ettention::Vec3ui& centerVertexCoord, std::vector<Material*>& materials);
    ProblemFragment(ettention::Vec3ui& centerVertexCoord);
    ~ProblemFragment();

    void setMaterial(unsigned int index, Material& mat);
    void setMaterial(unsigned int index, Material* mat);

    const ProblemFragmentKey& key() const;
    bool containsMixedMaterials() const;

    REAL mu(unsigned int cell) const;
    REAL lambda(unsigned int cell) const;

private:
    const ettention::Vec3ui m_centerVertexCoord;
    std::vector<Material*> m_materials;
    ProblemFragmentKey m_key;

};
