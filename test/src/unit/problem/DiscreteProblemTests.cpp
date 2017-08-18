#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "problem/DiscreteProblem.h"
#include "helpers/Templates.h"

class DiscreteProblemTests : public ::testing::Test {

public:
    DiscreteProblemTests() {}
    ~DiscreteProblemTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(DiscreteProblemTests, Initialization) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();

    bool didInit = true;
    for (int i = 0; i < 8; i++) {
        didInit = didInit && *problem.getMaterial(i) == Templates::Mat.STEEL;
    }
    EXPECT_TRUE(didInit) << "Materials were not initialized properly";
}

TEST_F(DiscreteProblemTests, BoundsHandling) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();

    Material* outOfBounds = problem.getMaterial(ettention::Vec3ui(3,3,3));
    EXPECT_TRUE(*outOfBounds == Material::EMPTY) << "Out of bounds coordinate did not return EMPTY material";

    try {
        problem.mapToVoxelIndex(ettention::Vec3ui(3, 3, 3));
        FAIL() << "Expected out of bounds voxel coordinate to throw an exception";
    }
    catch (const std::invalid_argument&) {}
    catch (...) {
        FAIL() << "Expected invalid argument exception";
    }

    unsigned int vertexIndex = problem.mapToVertexIndex(ettention::Vec3ui(2, 2, 2));
    EXPECT_EQ(vertexIndex, 26) << "Expected 2x2x2 problem to return vertex index 26 at vertex coordinate (2,2,2)";

    try {
        problem.mapToVertexIndex(ettention::Vec3ui(4, 3, 3));
        FAIL() << "Expected out of bounds vertex coordinate to throw an exception";
    }
    catch (const std::invalid_argument&) {}
    catch (...) {
        FAIL() << "Expected invalid argument exception";
    }

}

TEST_F(DiscreteProblemTests, ProblemFragmentExtraction) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();

    ProblemFragment fragment = problem.extractLocalProblem(ettention::Vec3ui(1, 1, 1));
    MaterialConfiguration actualKey = fragment.getMaterialConfiguration();
    std::vector<Material*> expected(8, &Templates::Mat.STEEL);
    MaterialConfiguration expectedKey(&expected);
    EXPECT_TRUE(actualKey == expectedKey) << "Expected fragment centered at (1,1,1) to be all STEEL";

    ProblemFragment fragment2 = problem.extractLocalProblem(ettention::Vec3ui(1,1,0));
    MaterialConfiguration actualKey2 = fragment2.getMaterialConfiguration();
    std::vector<Material*> expected2(8, &Templates::Mat.EMPTY);
    expected2[4] = expected2[5] = expected2[6] = expected2[7] = &Templates::Mat.STEEL;
    MaterialConfiguration expectedKey2(&expected2);
    EXPECT_TRUE(actualKey2 == expectedKey2) << "Fragment at (1,1,0) did not match expected result";
}

TEST_F(DiscreteProblemTests, DirichletBoundaryPlacement) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();

    problem.setDirichletBoundary(0, DirichletBoundary(DirichletBoundary::FIXED_ALL));
    problem.setDirichletBoundary(ettention::Vec3ui(2, 2, 2), DirichletBoundary(DirichletBoundary::FIXED_Z));

    DirichletBoundary actual = problem.getDirichletBoundaryAtVertex(ettention::Vec3ui(0, 0, 0));
    EXPECT_TRUE(actual == DirichletBoundary(DirichletBoundary::FIXED_ALL));

    actual = problem.getDirichletBoundaryAtVertex(ettention::Vec3ui(2, 2, 2));
    EXPECT_TRUE(actual == DirichletBoundary(DirichletBoundary::FIXED_Z));

    problem.setDirichletBoundary(0, DirichletBoundary(DirichletBoundary::NONE));
    actual = problem.getDirichletBoundaryAtVertex(ettention::Vec3ui(0, 0, 0));
    EXPECT_TRUE(actual == DirichletBoundary(DirichletBoundary::NONE));

    actual = problem.getDirichletBoundaryAtVertex(ettention::Vec3ui(1, 1, 1));
    EXPECT_TRUE(actual == DirichletBoundary(DirichletBoundary::NONE)) << "Expected vertex with no boundary condition to return DirichletBoundary::NONE";
}

TEST_F(DiscreteProblemTests, DirichletBoundaryApplication) {
    DiscreteProblem problem = Templates::Problem::STEEL_3_3_3();

    problem.setDirichletBoundary(ettention::Vec3ui(1, 0, 0), DirichletBoundary(DirichletBoundary::FIXED_ALL));

    //These two fragments would have the same MaterialConfiguration if it weren't for the boundary condition on the first one
    ProblemFragment fragment = problem.extractLocalProblem(ettention::Vec3ui(1, 0, 0));
    ProblemFragment nonBoundaryFragment = problem.extractLocalProblem(ettention::Vec3ui(2, 0, 0));

    EXPECT_FALSE(fragment.getMaterialConfiguration() == nonBoundaryFragment.getMaterialConfiguration()) << "Expected boundary condition to force a new material configuration";
}

