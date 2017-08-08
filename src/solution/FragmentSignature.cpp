#include <stdafx.h>
#include "FragmentSignature.h"

const size_t FragmentSignature::SizeInBytes = sizeof(REAL) * 9 * 27;

FragmentSignature::FragmentSignature(unsigned short id) : 
    m_id(id)
{
    m_matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

FragmentSignature::FragmentSignature() :
    m_id(USHRT_MAX)
{
    m_matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

FragmentSignature::~FragmentSignature() {

}

unsigned short FragmentSignature::getId() {
    return m_id;
}

Matrix3x3* FragmentSignature::getMatrices() {
    return m_matrices.data();
}

const Matrix3x3* FragmentSignature::getLHS() const {
    return &m_matrices[13];
}

const Matrix3x3* FragmentSignature::getRHS(unsigned int nodeIndex) const {
    return &m_matrices[nodeIndex];
}

void FragmentSignature::serialize(void* destination) const {
    Matrix3x3* dest = (Matrix3x3*)destination;

    for (unsigned int i = 0; i < 27; i++) {
        m_matrices[i].serialize(&dest[i]);
    }
}

void FragmentSignature::setId(unsigned short id) {
    m_id = id;
}

void FragmentSignature::setLHS(Matrix3x3& lhs) {
    m_matrices[13] = Matrix3x3(lhs);
}

void FragmentSignature::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    m_matrices[nodeIndex] = Matrix3x3(rhs);
}
