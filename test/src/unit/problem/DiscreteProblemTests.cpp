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
    ettention::Vec3ui size(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);

    DiscreteProblem problem(size, voxelSize);
    int numCells = 8;

    for (int i = 0; i < numCells; i++) {
        problem.setMaterial(i, Templates::Mat::STEEL);
    }

    bool didInit = true;
    for (int i = 0; i < numCells; i++) {
        didInit = didInit && *problem.getMaterial(i) == Templates::Mat::STEEL;
    }
    EXPECT_TRUE(didInit) << "Materials were not initialized properly";
}

TEST_F(DiscreteProblemTests, BoundsHandling) {
    ettention::Vec3ui size(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);

    DiscreteProblem problem(size, voxelSize);
    int numCells = 8;

    for (int i = 0; i < numCells; i++) {
        problem.setMaterial(i, Templates::Mat::STEEL);
    }

    Material* outOfBounds = problem.getMaterial(ettention::Vec3ui(3,3,3));
    EXPECT_TRUE(*outOfBounds == Material::EMPTY) << "Out of bounds coordinate did not return EMPTY material";

    try {
        problem.mapToIndex(ettention::Vec3ui(3, 3, 3));
        FAIL() << "Expected out of bounds coordinate to throw an exception";
    }
    catch (const std::invalid_argument&) {}
    catch (...) {
        FAIL() << "Expected invalid argument exception";
    }

}

TEST_F(DiscreteProblemTests, ProblemFragmentExtraction) {
    ettention::Vec3ui size(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);

    DiscreteProblem problem(size, voxelSize);
    int numCells = 8;

    for (int i = 0; i < numCells; i++) {
        problem.setMaterial(i, Templates::Mat::STEEL);
    }

    ProblemFragment fragment = problem.extractLocalProblem(ettention::Vec3ui(1, 1, 1));
    ProblemFragmentKey actualKey = fragment.key();
    std::vector<Material*> expected(8, &Templates::Mat::STEEL);
    ProblemFragmentKey expectedKey(&expected);
    EXPECT_TRUE(actualKey == expectedKey) << "Expected fragment centered at (1,1,1) to be all STEEL";

    ProblemFragment fragment2 = problem.extractLocalProblem(ettention::Vec3ui(1,1,0));
    ProblemFragmentKey actualKey2 = fragment2.key();
    std::vector<Material*> expected2(8, &Templates::Mat::EMPTY);
    expected2[4] = expected2[5] = expected2[6] = expected2[7] = &Templates::Mat::STEEL;
    ProblemFragmentKey expectedKey2(&expected2);
    EXPECT_TRUE(actualKey2 == expectedKey2) << "Fragment at (1,1,0) did not match expected result";

}
