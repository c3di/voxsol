#include <stdafx.h>
#include "MatrixStore.h"

MatrixStore::MatrixStore() {
    m_rhs = std::vector<Matrix3x3>(27, Matrix3x3::identity);
    m_rhs[13] = Matrix3x3::identity;
}

MatrixStore::~MatrixStore() {

}

void MatrixStore::setLHS(Matrix3x3& lhs) {
    m_lhs = Matrix3x3(lhs);
}

void MatrixStore::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    m_rhs[nodeIndex] = Matrix3x3(rhs);
}
