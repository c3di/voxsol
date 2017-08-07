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
    Solution sol(problem);

    const std::vector<unsigned short>* signatureIds = sol.getSignatureIds();
    ASSERT_EQ(signatureIds->size(), 27);

    sol.precomputeMatrices();
    const std::vector<FragmentSignature>* fSigs = sol.getFragmentSignatures();
    EXPECT_EQ(fSigs->size(), 27) << "Expected all vertices to have a unique fragment signature";

    // Since every vertex in this problem has a unique material configuration its signature ID should be the same as its index
    signatureIds = sol.getSignatureIds();
    for (unsigned short i = 0; i < 27; i++) {
        if (signatureIds->at(i) != i) {
            FAIL() << "Vertex " << i << " signature id (" << signatureIds->at(i) << ") did not match expected value " << i;
        }
    }

    ProblemFragment frag = problem.extractLocalProblem(ettention::Vec3ui(1, 1, 1));
    ProblemFragmentKey fragKey = frag.key();
    unsigned short fragId = sol.getSignatureIdForKey(fragKey);
    EXPECT_EQ(fragId, 13) << "Expected problem fragment for center vertex to have fragment id 13";
}

TEST_F(SolutionTests, ConsistencyAfterPrecompute) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();
    SolutionInspector sol(problem);
    std::string errMessage;

    ASSERT_TRUE(sol.solutionDimensionsMatchProblem(errMessage)) << errMessage;

    sol.precomputeMatrices();
    
    ASSERT_TRUE(sol.allVerticesHaveValidSignatureId(errMessage)) << errMessage;
    ASSERT_TRUE(sol.fragmentSignatureIdsMatchPositionInVector(errMessage)) << errMessage;
    ASSERT_TRUE(sol.allFragmentSignaturesInitialized(errMessage)) << errMessage;
}
