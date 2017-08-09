#pragma once
#include <stdafx.h>
#include <vector>
#include "math/Matrix3x3.h"

class MaterialConfigurationEquations {

public:
    MaterialConfigurationEquations();
    MaterialConfigurationEquations(ConfigId id);
    ~MaterialConfigurationEquations();

    const static size_t SizeInBytes;

    void setLHS(Matrix3x3& lhs);
    void setRHS(unsigned int nodeIndex, Matrix3x3& rhs);
    void setId(ConfigId id);
    ConfigId getId();
    Matrix3x3* getMatrices();

    bool isInitialized();
    void serialize(void* destination) const;

    const Matrix3x3* getLHS() const;
    const Matrix3x3* getRHS(unsigned int nodeIndex) const;

private:
    ConfigId id;
    std::vector<Matrix3x3> matrices; 
    
};
