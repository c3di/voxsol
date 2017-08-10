#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>

#include "libmmv/math/Vec3.h"
#include "libmmv/math/Matrix3x3.h"


class Matrix3x3 : public ettention::Matrix3x3<REAL>
{
public:
    Matrix3x3();
    Matrix3x3(REAL c0x, REAL c0y, REAL c0z, REAL c1x, REAL c1y, REAL c1z, REAL c2x, REAL c2y, REAL c2z);
    Matrix3x3(ettention::Vec3<REAL> column0, ettention::Vec3<REAL> column1, ettention::Vec3<REAL> column2);
    Matrix3x3(const std::vector<REAL>& values);
    Matrix3x3(const Matrix3x3& other);

    static const Matrix3x3 identity;
    const static size_t SizeInBytes;
    void serialize(void* destination) const;
};
