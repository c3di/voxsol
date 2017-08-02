#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"
#include "helpers/Templates.h"

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

    std::vector<int> fragmentIds = sol.getSignatureIds();
    ASSERT_EQ(fragmentIds.size(), 27);

    sol.precomputeMatrices();
    std::vector<MatrixStore> mStore = sol.getMatrixStore();
    EXPECT_EQ(mStore.size(), 27) << "Expected all vertices to have a unique material configuration";

    // Since every vertex has a unique material configuration its fragment ID should be the same as its index
    fragmentIds = sol.getSignatureIds();
    for (unsigned int i = 0; i < 27; i++) {
        if (fragmentIds[i] != i) {
            FAIL() << "Vertex " << i << " fragment id (" << fragmentIds[i] << ") did not match expected value " << i;
        }
    }

    ProblemFragment frag = problem.extractLocalProblem(ettention::Vec3ui(1, 1, 1));
    ProblemFragmentKey fragKey = frag.key();
    unsigned int fragId = sol.getSignatureIdForKey(fragKey);
    EXPECT_EQ(fragId, 13) << "Expected problem fragment for center vertex to have fragment id 13";
}
