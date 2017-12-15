#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>

#include "libmmv/math/Vec3.h"
#include "libmmv/math/Matrix3x3.h"

#define CLOSE_EQ_EPSILON std::numeric_limits<REAL>::epsilon

class Matrix3x3 : public libmmv::Matrix3x3<REAL>
{
public:
    Matrix3x3();
    Matrix3x3(REAL c0x, REAL c0y, REAL c0z, REAL c1x, REAL c1y, REAL c1z, REAL c2x, REAL c2y, REAL c2z);
    Matrix3x3(libmmv::Vec3<REAL> column0, libmmv::Vec3<REAL> column1, libmmv::Vec3<REAL> column2);
    Matrix3x3(const std::vector<REAL>& values);
    Matrix3x3(const Matrix3x3& other);
    Matrix3x3(const libmmv::Matrix3x3<REAL>& other);

    static const Matrix3x3 identity;
    const static size_t SizeInBytes;
    void serialize(void* destination) const;

    REAL conditionNumber() const;
};
