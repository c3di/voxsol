#pragma once
#include <stdafx.h>
#include <vector>
#include "math/Matrix3x3.h"

class FragmentSignature {

public:
    FragmentSignature();
    FragmentSignature(unsigned short id);
    ~FragmentSignature();

    const static size_t SizeInBytes;

    void setLHS(Matrix3x3& lhs);
    void setRHS(unsigned int nodeIndex, Matrix3x3& rhs);
    void setId(unsigned short id);
    unsigned short getId();
    Matrix3x3* getMatrices();

    void serialize(void* destination) const;

    const Matrix3x3* getLHS() const;
    const Matrix3x3* getRHS(unsigned int nodeIndex) const;

private:
    unsigned short m_id;
    std::vector<Matrix3x3> m_matrices; 
    
};
