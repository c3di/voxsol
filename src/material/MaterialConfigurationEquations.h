#pragma once
#include "math/Matrix3x3.h"
#include "problem/DirichletBoundary.h"
#include <vector>

class MaterialConfigurationEquations {

public:
    MaterialConfigurationEquations();
    MaterialConfigurationEquations(ConfigId id);
    MaterialConfigurationEquations(ConfigId id, DirichletBoundary& dirichletBoundaryCondition);
    ~MaterialConfigurationEquations();

    const static size_t SizeInBytes;

    void setLHSInverse(Matrix3x3& lhs);
    void setRHS(unsigned int nodeIndex, Matrix3x3& rhs);
    void setId(ConfigId id);

    ConfigId getId();
    Matrix3x3* getMatrices();

    bool isInitialized();
    void serialize(void* destination) const;

    const Matrix3x3* getLHSInverse() const;
    const Matrix3x3* getRHS(unsigned int nodeIndex) const;

private:
    ConfigId id;
    DirichletBoundary dirichletBoundaryCondition;
    std::vector<Matrix3x3> matrices; 
    
};
