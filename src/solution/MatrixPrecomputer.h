#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <libmmv/math/Vec3.h>
#include <libmmv/math/Matrix3x3.h>
#include "equations/LinearBaseIntegrals.h"
#include "equations/QuadraticBaseIntegrals.h"
#include "solution/MatrixStore.h"
#include "problem/ProblemFragment.h"

class MatrixPrecomputer {
public:

    MatrixPrecomputer(ettention::Vec3<REAL> voxelSize);
    ~MatrixPrecomputer();

    MatrixStore computeMatrixStoreForFragment(const ProblemFragment& fragment) const;

private:
    const ettention::Vec3<REAL> m_voxelSize;
    const LinearBaseIntegrals m_linearIntegrals;
    const QuadraticBaseIntegrals m_quadIntegrals;

    void computeLHS(const ProblemFragment& fragment, MatrixStore& store, const BaseIntegrals* integrals) const;
    void computeRHS(const ProblemFragment& fragment, MatrixStore& store, const BaseIntegrals* integrals) const;

    void computeRHSForNode(unsigned int nodeIndex, const ProblemFragment& fragment, MatrixStore& store, const BaseIntegrals* integrals) const;
};
