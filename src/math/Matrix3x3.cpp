#include "stdafx.h"
#include "math/Matrix3x3.h"

const size_t Matrix3x3::SizeInBytes = sizeof(REAL) * 9;
const Matrix3x3 Matrix3x3::identity = Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);

Matrix3x3::Matrix3x3()
{
}

Matrix3x3::Matrix3x3(const std::vector<REAL>& values) :
    libmmv::Matrix3x3<REAL>(values)
{
    
}

Matrix3x3::Matrix3x3(REAL c0x, REAL c0y, REAL c0z, REAL c1x, REAL c1y, REAL c1z, REAL c2x, REAL c2y, REAL c2z) :
    libmmv::Matrix3x3<REAL>(c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z)
{
    
}

Matrix3x3::Matrix3x3(libmmv::Vec3<REAL> column0, libmmv::Vec3<REAL> column1, libmmv::Vec3<REAL> column2) :
    libmmv::Matrix3x3<REAL>(column0, column1, column2)
{
    
}

Matrix3x3::Matrix3x3(const libmmv::Matrix3x3<REAL>& other) :
    libmmv::Matrix3x3<REAL>(other)
{

}

Matrix3x3::Matrix3x3(const Matrix3x3& other) 
{
    memcpy(values, other.values, 9 * sizeof(REAL));
}

void Matrix3x3::serialize(void* destination) const {
    memcpy(destination, &values[0], Matrix3x3::SizeInBytes);
}

REAL Matrix3x3::conditionNumber() const {
    REAL max = std::max(std::max(values[0], values[4]), values[8]);
    REAL min = std::min(std::min(values[0], values[4]), values[8]);
    return max / min;
}

