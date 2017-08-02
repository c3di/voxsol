#pragma once
#include <stdafx.h>
#include <vector>
#include "math/Matrix3x3.h"

class MatrixStore {

public:
    MatrixStore();
    MatrixStore(int id);
    ~MatrixStore();

    void setLHS(Matrix3x3& lhs);
    void setRHS(unsigned int nodeIndex, Matrix3x3& rhs);
    int getId();
    void setId(int id);

    inline const Matrix3x3* getLHS() const {
        return &m_lhs;
    }
    inline const Matrix3x3* getRHS(unsigned int nodeIndex) const {
        return &m_rhs[nodeIndex];
    }

private:
    Matrix3x3 m_lhs;
    std::vector<Matrix3x3> m_rhs; 
    int m_id;
};
