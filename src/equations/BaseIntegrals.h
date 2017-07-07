#pragma once
#include <stdafx.h>
#include "libmmv/math/Vec3.h"

#define MAP(A, func, deriv, testFunc, cell) A[(func)*3*3*8 + (deriv)*3*8 + (testFunc)*8 + (cell)]

class BaseIntegrals {
public:

    BaseIntegrals(ettention::Vec3f& voxelSize) :
        dx(voxelSize.x),
        dy(voxelSize.y),
        dz(voxelSize.z)
    {

    }

    BaseIntegrals(ettention::Vec3d& voxelSize) :
        dx(voxelSize.x),
        dy(voxelSize.y),
        dz(voxelSize.z)
    {

    }

    inline REAL* data() const {
        return (REAL*)&m_data;
    }
    inline REAL value(unsigned int basisFunc, unsigned int deriv, unsigned int testFunc, unsigned int cell) const {
        return MAP(m_data, basisFunc, deriv, testFunc, cell);
    }

protected:
    const REAL dx;
    const REAL dy;
    const REAL dz;
    REAL m_data[27 * 3 * 3 * 8];

    virtual void init() = 0;
};
