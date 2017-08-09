#pragma once
#include <libmmv/math/Vec3.h>
#include "equations/LinearBaseIntegrals.h"
#include "equations/QuadraticBaseIntegrals.h"
#include "solution/FragmentSignature.h"
#include "problem/ProblemFragment.h"

class MatrixPrecomputer {
public:

    MatrixPrecomputer(ettention::Vec3<REAL> voxelSize);
    ~MatrixPrecomputer();

    void initializeSignatureForFragment(FragmentSignature* signature, const ProblemFragment& fragment) const;

private:
    const ettention::Vec3<REAL> voxelSize;
    const LinearBaseIntegrals linearIntegrals;
    const QuadraticBaseIntegrals quadIntegrals;

    void computeLHS(const ProblemFragment& fragment, FragmentSignature* signature, const BaseIntegrals* integrals) const;
    void computeRHS(const ProblemFragment& fragment, FragmentSignature* signature, const BaseIntegrals* integrals) const;

    void computeRHSForNode(unsigned int nodeIndex, const ProblemFragment& fragment, FragmentSignature* signature, const BaseIntegrals* integrals) const;
};
