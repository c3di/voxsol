#include <stdafx.h>
#include "MaterialConfigurationEquationsFactory.h"
#include "material/MaterialConfigurationEquations.h"
#include "problem/ProblemFragment.h"
#include "math/Matrix3x3.h"


MaterialConfigurationEquationsFactory::MaterialConfigurationEquationsFactory(ettention::Vec3<REAL> voxelSize) :
    voxelSize(voxelSize),
    linearIntegrals(voxelSize),
    quadIntegrals(voxelSize)
{
    
}

MaterialConfigurationEquationsFactory::~MaterialConfigurationEquationsFactory() {

}

void MaterialConfigurationEquationsFactory::initializeEquationsForFragment(MaterialConfigurationEquations* equations, const ProblemFragment& fragment) const {
    const BaseIntegrals* integrals;
    
    //Quadratic integrals can only be used for fragments with uniform materials, mixed materials will cause
    //the solution to diverge
    if (fragment.containsMixedMaterials()) {
        integrals = &linearIntegrals;
    }
    else {
        integrals = &quadIntegrals;
    }

    computeLHS(fragment, equations, integrals);
    computeRHS(fragment, equations, integrals);
    setNeumannBoundary(fragment, equations);
}

void MaterialConfigurationEquationsFactory::computeLHS(const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const {
    REAL* fullIntegralLHS = new REAL[21]();

    for (unsigned int cell = 0; cell < 8; cell++) {
        REAL mu = fragment.mu(cell);
        REAL lambda = fragment.lambda(cell);

        fullIntegralLHS[0] += (lambda + 2.0 * mu) * integrals->value(13,0,0,cell);
        fullIntegralLHS[1] += lambda * integrals->value(13,1,0,cell);
        fullIntegralLHS[2] += lambda * integrals->value(13,2,0,cell);
        fullIntegralLHS[3] += mu * integrals->value(13,1,1,cell);
        fullIntegralLHS[4] += mu * integrals->value(13,0,1,cell);

        fullIntegralLHS[5] += mu * integrals->value(13,2,2,cell);
        fullIntegralLHS[6] += mu * integrals->value(13,0,2,cell);
        fullIntegralLHS[7] += (lambda + 2.0 * mu) * integrals->value(13,1,1,cell);
        fullIntegralLHS[8] += lambda * integrals->value(13,0,1,cell);
        fullIntegralLHS[9] += lambda * integrals->value(13,2,1,cell);

        fullIntegralLHS[10] += mu * integrals->value(13,1,0,cell);
        fullIntegralLHS[11] += mu * integrals->value(13,0,0,cell);
        fullIntegralLHS[12] += mu * integrals->value(13,2,2,cell);
        fullIntegralLHS[13] += mu * integrals->value(13,1,2,cell);
        fullIntegralLHS[14] += (lambda + 2.0 * mu) * integrals->value(13,2,2,cell);

        fullIntegralLHS[15] += lambda * integrals->value(13,0,2,cell);
        fullIntegralLHS[16] += lambda * integrals->value(13,1,2,cell);
        fullIntegralLHS[17] += mu * integrals->value(13,2,0,cell);
        fullIntegralLHS[18] += mu * integrals->value(13,0,0,cell);
        fullIntegralLHS[19] += mu * integrals->value(13,2,1,cell);

        fullIntegralLHS[20] += mu * integrals->value(13,1,1,cell);
    }
    ettention::Vec3<REAL> col1(
        fullIntegralLHS[0] + fullIntegralLHS[3] + fullIntegralLHS[5], 
        fullIntegralLHS[1] + fullIntegralLHS[4], 
        fullIntegralLHS[2] + fullIntegralLHS[6]
    );
    ettention::Vec3<REAL> col2(
        fullIntegralLHS[8] + fullIntegralLHS[10],
        fullIntegralLHS[7] + fullIntegralLHS[11] + fullIntegralLHS[12],
        fullIntegralLHS[9] + fullIntegralLHS[13]
    );
    ettention::Vec3<REAL> col3(
        fullIntegralLHS[15] + fullIntegralLHS[17],
        fullIntegralLHS[16] + fullIntegralLHS[19],
        fullIntegralLHS[14] + fullIntegralLHS[18] + fullIntegralLHS[20]
    );
    Matrix3x3 lhs(col1, col2, col3);
    checkMatrixConditionNumber(lhs);
    lhs = Matrix3x3(lhs.inverse());

    applyDirichletBoundaryConditionsToLHS(lhs, fragment);

    equations->setLHSInverse(lhs);

    delete[] fullIntegralLHS;
}

void MaterialConfigurationEquationsFactory::checkMatrixConditionNumber(const Matrix3x3& mat) const {
    REAL condition = mat.conditionNumber();
    if (condition > CONDITION_NUMBER_MAX) {
        std::cerr << "Warning: Inverting an ill-conditioned matrix: Cond = " << condition << " for matrix " << mat << std::endl;
    }
}

void MaterialConfigurationEquationsFactory::applyDirichletBoundaryConditionsToLHS(Matrix3x3& lhsInverse, const ProblemFragment& fragment) const {
    DirichletBoundary condition = fragment.getDirichletBoundaryCondition();
    if (condition.isXFixed()) {
        // Set first row to 0's, when multiplied with RHS vector later the x component will always be 0 => fixed. Same strategy for y and z below
        lhsInverse = Matrix3x3(0, lhsInverse.at(0, 1), lhsInverse.at(0, 2), 0, lhsInverse.at(1, 1), lhsInverse.at(1, 2), 0, lhsInverse.at(2, 1), lhsInverse.at(2, 2));
    }
    if (condition.isYFixed()) {
        lhsInverse = Matrix3x3(lhsInverse.at(0, 0), 0, lhsInverse.at(0, 2), lhsInverse.at(1, 0), 0, lhsInverse.at(1, 2), lhsInverse.at(2, 0), 0, lhsInverse.at(2, 2));
    }
    if (condition.isZFixed()) {
        lhsInverse = Matrix3x3(lhsInverse.at(0, 0), lhsInverse.at(0, 1), 0, lhsInverse.at(1, 0), lhsInverse.at(1, 1), 0, lhsInverse.at(2, 0), lhsInverse.at(2, 1), 0);
    }
}

void MaterialConfigurationEquationsFactory::computeRHS(const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const {
    for (unsigned int nodeIndex = 0; nodeIndex < 27; nodeIndex++) {
        if (nodeIndex == 13) {
            //Don't need to calculate RHS for the center node
            continue;
        }
        computeRHSForNode(nodeIndex, fragment, equations, integrals);
    }
}

void MaterialConfigurationEquationsFactory::computeRHSForNode(unsigned int nodeIndex, const ProblemFragment& fragment, MaterialConfigurationEquations* equations, const BaseIntegrals* integrals) const {
    REAL* matrixRHS = new REAL[9](); //3x3 matrix in column-major

    for (int cell = 0; cell < 8; cell++) {
        double lambda = fragment.lambda(cell);
        //Column 1
        matrixRHS[0] += lambda * integrals->value(nodeIndex,0,0,cell);
        matrixRHS[1] += lambda * integrals->value(nodeIndex,0,1,cell);
        matrixRHS[2] += lambda * integrals->value(nodeIndex,0,2,cell);
        //Column 2
        matrixRHS[3] += lambda * integrals->value(nodeIndex,1,0,cell);
        matrixRHS[4] += lambda * integrals->value(nodeIndex,1,1,cell);
        matrixRHS[5] += lambda * integrals->value(nodeIndex,1,2,cell);
        //Column 3
        matrixRHS[6] += lambda * integrals->value(nodeIndex,2,0,cell);
        matrixRHS[7] += lambda * integrals->value(nodeIndex,2,1,cell);
        matrixRHS[8] += lambda * integrals->value(nodeIndex,2,2,cell);

        double mu = fragment.mu(cell);
        matrixRHS[0] += mu * (2 * integrals->value(nodeIndex,0,0,cell) + integrals->value(nodeIndex,1,1,cell) + integrals->value(nodeIndex,2,2,cell));
        matrixRHS[1] += mu * integrals->value(nodeIndex,1,0,cell);
        matrixRHS[2] += mu * integrals->value(nodeIndex,2,0,cell);

        matrixRHS[3] += mu * integrals->value(nodeIndex,0,1,cell);
        matrixRHS[4] += mu * (2 * integrals->value(nodeIndex,1,1,cell) + integrals->value(nodeIndex,0,0,cell) + integrals->value(nodeIndex,2,2,cell));
        matrixRHS[5] += mu * integrals->value(nodeIndex,2,1,cell);

        matrixRHS[6] += mu * integrals->value(nodeIndex,0,2,cell);
        matrixRHS[7] += mu * integrals->value(nodeIndex,1,2,cell);
        matrixRHS[8] += mu * (2 * integrals->value(nodeIndex,2,2,cell) + integrals->value(nodeIndex,0,0,cell) + integrals->value(nodeIndex,1,1,cell));
    }
    
    Matrix3x3 rhs(matrixRHS[0], matrixRHS[1], matrixRHS[2], matrixRHS[3], matrixRHS[4], matrixRHS[5], matrixRHS[6], matrixRHS[7], matrixRHS[8]);
    equations->setRHS(nodeIndex, rhs);

    delete[] matrixRHS;
}

void MaterialConfigurationEquationsFactory::setNeumannBoundary(const ProblemFragment& fragment, MaterialConfigurationEquations* equations) const {
    equations->setNeumannBoundaryCondition(fragment.getNeumannBoundaryCondition());
}

