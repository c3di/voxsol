#include "stdafx.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "helpers/Templates.h"
#include "helpers/SolutionInspector.h"
#include "problem/boundaryconditions/NeumannBoundary.h"

class SolutionTests : public ::testing::Test {

public:
    SolutionTests() {}
    ~SolutionTests() {}

    bool closeEqual(const Matrix3x3& a, const Matrix3x3& b) {
        for (unsigned int i = 0; i < 9; i++) {
            const testing::internal::FloatingPoint<REAL> lhs(a.at(i % 3, i / 3)), rhs(b.at(i % 3, i / 3));
            if (!lhs.AlmostEquals(rhs)) {
                return false;
            }
        }
        return true;
    }

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(SolutionTests, BoundsHandling) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();
    Solution solution(problem);

    try {
        solution.mapToIndex(libmmv::Vec3ui(3, 3, 3));
        FAIL() << "Expected out of bounds coordinate to throw an exception";
    }
    catch (const std::invalid_argument&) {}
    catch (...) {
        FAIL() << "Expected invalid argument exception";
    }
}

TEST_F(SolutionTests, PrecomputeMatrices) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();
    SolutionInspector sol(problem);

    const std::vector<Vertex>* vertices = sol.getVertices();
    ASSERT_EQ(vertices->size(), 27);

    sol.computeMaterialConfigurationEquations();
    const std::vector<MaterialConfigurationEquations>* fSigs = sol.getMaterialConfigurationEquations();
    EXPECT_EQ(fSigs->size(), 27) << "Expected all vertices to have a unique set of material configuration equations";

    // Since every vertex in this problem has a unique material configuration its signature ID should be the same as its index
    vertices = sol.getVertices();
    for (ConfigId i = 0; i < 27; i++) {
        if (vertices->at(i).materialConfigId != i) {
            FAIL() << "Vertex " << i << " equation id (" << vertices->at(i).materialConfigId << ") did not match expected value " << i;
        }
    }

    ProblemFragment frag = problem.extractLocalProblem(libmmv::Vec3ui(1, 1, 1));
    ConfigId fragId = sol.getEquationIdForFragment(frag);
    EXPECT_EQ(fragId, 13) << "Expected problem fragment for center vertex to have material configuration equation id 13";
}

TEST_F(SolutionTests, ConsistencyAfterPrecompute) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();
    SolutionInspector sol(problem);
    std::string errMessage;

    ASSERT_TRUE(sol.solutionDimensionsMatchProblem(errMessage)) << errMessage;

    sol.computeMaterialConfigurationEquations();
    
    ASSERT_TRUE(sol.allVerticesHaveValidSignatureId(errMessage)) << errMessage;
    ASSERT_TRUE(sol.matConfigEquationIdsMatchPositionInVector(errMessage)) << errMessage;
    ASSERT_TRUE(sol.allMatConfigEquationsInitialized(errMessage)) << errMessage;
}

TEST_F(SolutionTests, DirichletBoundaryAppliedToLHS) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();

    problem.setDirichletBoundaryAtVertex(libmmv::Vec3ui(1, 0, 0), DirichletBoundary(DirichletBoundary::FIXED_ALL));
    problem.setDirichletBoundaryAtVertex(libmmv::Vec3ui(2, 0, 0), DirichletBoundary(DirichletBoundary::FIXED_X));

    SolutionInspector sol(problem);
    
    sol.computeMaterialConfigurationEquations();

    const std::vector<MaterialConfigurationEquations>* equations = sol.getMaterialConfigurationEquations();

    ProblemFragment allFixed = problem.extractLocalProblem(libmmv::Vec3ui(1, 0, 0));
    unsigned short eqId = sol.getEquationIdForFragment(allFixed);
    const MaterialConfigurationEquations allFixedEqns = equations->at(eqId);
    const Matrix3x3* allFixedLHS = allFixedEqns.getLHSInverse();
    Matrix3x3 allFixedExpected(0, 0, 0, 0, 0, 0, 0, 0, 0);

    EXPECT_TRUE(closeEqual(*allFixedLHS, allFixedExpected)) << "Expected LHS matrix for ALL_FIXED to be all zeros";

    ProblemFragment xFixed = problem.extractLocalProblem(libmmv::Vec3ui(2, 0, 0));
    eqId = sol.getEquationIdForFragment(xFixed);
    const MaterialConfigurationEquations xFixedEqns = equations->at(eqId);
    const Matrix3x3* xFixedLHS = xFixedEqns.getLHSInverse();
    //Note: first row is all zeroes => x component zero when multiplied with RHS vector => x is fixed
    Matrix3x3 xFixedExpected(0.0, asREAL(6.2308614032751929e-12), asREAL(6.2308614032751929e-12), 0.0, asREAL(2.4508054852882432e-11), asREAL(-6.2308614032751929e-12), 0.0, asREAL(-6.2308614032751929e-12), asREAL(2.4508054852882432e-11));

    EXPECT_TRUE(closeEqual(*xFixedLHS, xFixedExpected)) << "Expected LHS matrix for X_FIXED to have zeroes in first row";
}

TEST_F(SolutionTests, NeumannBoundaryAppliedToEquations) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();

    problem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(1, 0, 0), NeumannBoundary(libmmv::Vec3<REAL>(9999, 0, 0)));
    problem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(2, 0, 0), NeumannBoundary(libmmv::Vec3<REAL>(100, 100, 100)));

    SolutionInspector sol(problem);

    sol.computeMaterialConfigurationEquations();

    const std::vector<MaterialConfigurationEquations>* equations = sol.getMaterialConfigurationEquations();

    ProblemFragment stressInX = problem.extractLocalProblem(libmmv::Vec3ui(1, 0, 0));
    unsigned short eqId = sol.getEquationIdForFragment(stressInX);
    const MaterialConfigurationEquations stressInXEqns = equations->at(eqId);

    EXPECT_TRUE(stressInXEqns.getNeumannBoundaryCondition()->stress.x == 9999) << "Expected material config equation to store the right neumann boundary stress";

    ProblemFragment uniformStress = problem.extractLocalProblem(libmmv::Vec3ui(2, 0, 0));
    eqId = sol.getEquationIdForFragment(uniformStress);
    const MaterialConfigurationEquations uniformStressEqns = equations->at(eqId);
    libmmv::Vec3<REAL> stress = uniformStressEqns.getNeumannBoundaryCondition()->stress;

    EXPECT_TRUE(stress.x == 100 && stress.y == 100 && stress.z == 100) << "Expected material config to store the right neumann boundary stress";
}
