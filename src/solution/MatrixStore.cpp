#include <stdafx.h>
#include "MatrixStore.h"

MatrixStore::MatrixStore(int id) : 
    m_id(id)
{
    m_rhs = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

MatrixStore::MatrixStore() :
    m_id(-1)
{
    m_rhs = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

MatrixStore::~MatrixStore() {

}

int MatrixStore::getId() {
    return m_id;
}

void MatrixStore::setId(int id) {
    m_id = id;
}

void MatrixStore::setLHS(Matrix3x3& lhs) {
    m_lhs = Matrix3x3(lhs);
}

void MatrixStore::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    m_rhs[nodeIndex] = Matrix3x3(rhs);
}
