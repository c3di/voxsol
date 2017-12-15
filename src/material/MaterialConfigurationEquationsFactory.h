#pragma once
#include <libmmv/math/Vec3.h>
#include "equations/LinearBaseIntegrals.h"
#include "equations/QuadraticBaseIntegrals.h"

//Somewhat arbitrary upper limit, derived from the general rule of thumb: cond(A) = 10^k -> up to k inaccurate digits
//Condition numbers larger than 10^3 may not cause large inaccuracies with float/double precision but can slow down convergence
#define CONDITION_NUMBER_MAX (10^3)

class MaterialConfigurationEquations;
class ProblemFragment;
class Matrix3x3;

class MaterialConfigurationEquationsFactory {
public:

    MaterialConfigurationEquationsFactory(libmmv::Vec3<REAL> voxelSize);
    ~MaterialConfigurationEquationsFactory();

    void initializeEquationsForFragment(MaterialConfigurationEquations* equations, const ProblemFragment& fragment) const;

private:
    const libmmv::Vec3<REAL> voxelSize;
    const LinearBaseIntegrals linearIntegrals;
    const QuadraticBaseIntegrals quadIntegrals;

    void computeLHS(const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const;
    void applyDirichletBoundaryConditionsToLHS(Matrix3x3& lhsInverse, const ProblemFragment& fragment) const;

    void computeRHS(const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const;
    void computeRHSForNode(unsigned int nodeIndex, const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const;

    void setNeumannBoundary(const ProblemFragment& fragment, MaterialConfigurationEquations* equations) const;

    void checkMatrixConditionNumber(const Matrix3x3& mat) const;
};
