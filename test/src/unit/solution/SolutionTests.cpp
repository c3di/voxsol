#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "helpers/Templates.h"
#include "helpers/SolutionInspector.h"

class SolutionTests : public ::testing::Test {

public:
    SolutionTests() {}
    ~SolutionTests() {}

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
        solution.mapToIndex(ettention::Vec3ui(3, 3, 3));
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

    const std::vector<ConfigId>* equationIds = sol.getMaterialConfigurationEquationIds();
    ASSERT_EQ(equationIds->size(), 27);

    sol.computeMaterialConfigurationEquations();
    const std::vector<MaterialConfigurationEquations>* fSigs = sol.getMaterialConfigurationEquations();
    EXPECT_EQ(fSigs->size(), 27) << "Expected all vertices to have a unique set of material configuration equations";

    // Since every vertex in this problem has a unique material configuration its signature ID should be the same as its index
    equationIds = sol.getMaterialConfigurationEquationIds();
    for (ConfigId i = 0; i < 27; i++) {
        if (equationIds->at(i) != i) {
            FAIL() << "Vertex " << i << " equation id (" << equationIds->at(i) << ") did not match expected value " << i;
        }
    }

    ProblemFragment frag = problem.extractLocalProblem(ettention::Vec3ui(1, 1, 1));
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
