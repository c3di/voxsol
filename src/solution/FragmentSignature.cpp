#include <stdafx.h>
#include "FragmentSignature.h"

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

void FragmentSignature::setId(unsigned short id) {
    m_id = id;
}

void FragmentSignature::setLHS(Matrix3x3& lhs) {
    m_matrices[13] = Matrix3x3(lhs);
}

void FragmentSignature::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    m_matrices[nodeIndex] = Matrix3x3(rhs);
}
