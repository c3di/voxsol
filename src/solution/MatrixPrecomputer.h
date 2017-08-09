#pragma once
#include <libmmv/math/Vec3.h>
#include "equations/LinearBaseIntegrals.h"
#include "equations/QuadraticBaseIntegrals.h"
#include "solution/MaterialConfigurationEquations.h"
#include "problem/ProblemFragment.h"

class MatrixPrecomputer {
public:

    MatrixPrecomputer(ettention::Vec3<REAL> voxelSize);
    ~MatrixPrecomputer();

    void initializeEquationsForFragment(MaterialConfigurationEquations* equations, const ProblemFragment& fragment) const;

private:
    const ettention::Vec3<REAL> voxelSize;
    const LinearBaseIntegrals linearIntegrals;
    const QuadraticBaseIntegrals quadIntegrals;

    void computeLHS(const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const;
    void computeRHS(const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const;

    void computeRHSForNode(unsigned int nodeIndex, const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const;
};
