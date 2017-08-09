#include <stdafx.h>
#include "FragmentSignature.h"

const size_t FragmentSignature::SizeInBytes = sizeof(REAL) * 9 * 27;

FragmentSignature::FragmentSignature(unsigned short id) : 
    id(id)
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

FragmentSignature::FragmentSignature() :
    id(USHRT_MAX)
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

FragmentSignature::~FragmentSignature() {

}

unsigned short FragmentSignature::getId() {
    return id;
}

Matrix3x3* FragmentSignature::getMatrices() {
    return matrices.data();
}

const Matrix3x3* FragmentSignature::getLHS() const {
    return &matrices[13];
}

const Matrix3x3* FragmentSignature::getRHS(unsigned int nodeIndex) const {
    return &matrices[nodeIndex];
}

void FragmentSignature::serialize(void* destination) const {
    Matrix3x3* dest = (Matrix3x3*)destination;

    for (unsigned int i = 0; i < 27; i++) {
        matrices[i].serialize(&dest[i]);
    }
}

void FragmentSignature::setId(unsigned short id) {
    this->id = id;
}

void FragmentSignature::setLHS(Matrix3x3& lhs) {
    matrices[13] = Matrix3x3(lhs);
}

void FragmentSignature::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    matrices[nodeIndex] = Matrix3x3(rhs);
}
