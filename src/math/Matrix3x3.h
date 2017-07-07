#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>

#include "libmmv/math/Vec3.h"


class Matrix3x3
{
public:
    Matrix3x3();
    Matrix3x3(REAL c0x, REAL c0y, REAL c0z, REAL c1x, REAL c1y, REAL c1z, REAL c2x, REAL c2y, REAL c2z);
    Matrix3x3(ettention::Vec3<REAL> column0, ettention::Vec3<REAL> column1, ettention::Vec3<REAL> column2);
    Matrix3x3(const std::vector<REAL>& values);
    Matrix3x3(const Matrix3x3& other);

    Matrix3x3& operator=(const Matrix3x3& other);

    Matrix3x3 inverse();
    REAL determinant();

    const REAL& at(unsigned int column, unsigned int row) const;
    REAL& at(unsigned int column, unsigned int row);

    bool operator==(const Matrix3x3& other) const;

    static const Matrix3x3 identity;
    static Matrix3x3 translationMatrix(ettention::Vec2<REAL>& translate);
    static Matrix3x3 scaleMatrix(REAL v);
    static Matrix3x3 rotationMatrix(REAL phi);

private:
    int indexOfMinor(int i, int droppedIndex);
    REAL determinantOfMinor(int x, int y);

    REAL values[9];
};


bool operator!=(const Matrix3x3& one, const Matrix3x3& other);

ettention::Vec2<REAL> multiply(const Matrix3x3& matrix, const ettention::Vec2<REAL>& vector);
ettention::Vec3<REAL> multiply(const Matrix3x3& matrix, const ettention::Vec3<REAL>& vector);
Matrix3x3 multiply(const Matrix3x3& a, const Matrix3x3& b);

std::istream &operator >> (std::istream& is, Matrix3x3& value);
std::ostream &operator<<(std::ostream& os, Matrix3x3 value);

