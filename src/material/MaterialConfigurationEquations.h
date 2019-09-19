#pragma once
#include "math/Matrix3x3.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include <vector>

class MaterialConfigurationEquations {

public:
    MaterialConfigurationEquations();
    MaterialConfigurationEquations(ConfigId id);
    ~MaterialConfigurationEquations();

    const static size_t SizeInBytes;

    void setLHS(Matrix3x3& lhs);
    void setLHSInverse(Matrix3x3& lhs);
    void setRHS(unsigned int nodeIndex, Matrix3x3& rhs);
    void setId(ConfigId id);
    void setNeumannBoundaryCondition(const NeumannBoundary& stress);
    void setDirichletBoundaryCondition(const DirichletBoundary& condition);

    ConfigId getId();
    Matrix3x3* getMatrices();

    bool isInitialized();
    void serialize(void* destination) const;

    const Matrix3x3* getLHSInverse() const;
    const Matrix3x3* getRHS(unsigned int nodeIndex) const;
    const NeumannBoundary* getNeumannBoundaryCondition() const;

private:
    ConfigId id;
    std::vector<Matrix3x3> matrices; 
    NeumannBoundary neumannBoundaryCondition;
    char dirichlet[3];
    
};
