#pragma once
#include <vector>
#include "libmmv/math/Vec3.h"
#include "problem/Material.h"

class ProblemFragment {
public:

    ProblemFragment(ettention::Vec3ui centerVertexCoord);
    ~ProblemFragment();

    void setMaterial(unsigned int index, Material& mat);
    void setMaterial(unsigned int index, Material* mat);
    std::string getMaterialConfiguration() const;
    bool containsMixedMaterials() const;

    inline REAL mu(unsigned int cell) const {
        return m_materials[cell]->m_mu;
    }

    inline REAL lambda(unsigned int cell) const {
        return m_materials[cell]->m_lambda;
    }

private:
    const ettention::Vec3ui m_centerVertexCoord;
    std::vector<Material*> m_materials;

};
