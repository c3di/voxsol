#include "stdafx.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"
#include "libmmv/math/Vec3.h"
#include "material/MaterialConfigurationEquationsFactory.h"
#include "material/MaterialConfigurationEquations.h"
#include "helpers/Templates.h"
#include "helpers/SolutionInspector.h"
#include "helpers/RHSMatricesReader.h"

class MCEFactoryTests : public ::testing::Test {

public:
    MCEFactoryTests() {}
    ~MCEFactoryTests() {}

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

// NOTE: Hard coded values come from the stomech prototype solver
// NOTE: When trying to replicate these values in the prototype make sure the voxel sizes match!

TEST_F(MCEFactoryTests, UniformSteel) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(0.5, 0.5, 0.5);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.STEEL);
    ProblemFragment fragment(coord, mats);

    // Will use quadratic equations since this fragment is made of a uniform material
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    const Matrix3x3* lhs = fragEquations.getLHSInverse();
    std::vector<REAL> lhsExpected({ asREAL(5.064935064935065E-12), 0, 0, 0, asREAL(5.064935064935065E-12), 0, 0, 0, asREAL(5.064935064935065E-12) });
    
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations.getRHS(1);
    std::vector<REAL> rhsExpected({asREAL(-1.1217948717948723E9), 0, 0, 0, asREAL(-1.794871794871795E10), asREAL(-1.6826923076923075E10), 0, asREAL(-1.6826923076923075E10),asREAL(-1.794871794871795E10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations.getRHS(9);
    std::vector<REAL> rhsExpected2({ asREAL(-1.794871794871795E10), asREAL(-1.6826923076923075E10), asREAL(0.0),
                                     asREAL(-1.6826923076923075E10), asREAL(-1.794871794871795E10), asREAL(0.0),
                                     asREAL(0.0), asREAL(0.0), asREAL(-1.1217948717948723E9) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));

    rhs = fragEquations.getRHS(22);
    std::vector<REAL> rhsExpected3({ asREAL(2.2435897435897438E10), asREAL(7.152557373046875E-7), asREAL(0.0),
                                     asREAL(7.152557373046875E-7), asREAL(2.2435897435897438E10), asREAL(-4.76837158203125E-7),
                                     asREAL(0.0), asREAL(4.76837158203125E-7), asREAL(-4.4871794871794876E10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected3)));
}

TEST_F(MCEFactoryTests, MixSteelNull) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.EMPTY);
    mats[4] = mats[5] = mats[6] = mats[7] = &Templates::Mat.STEEL;
 
    ProblemFragment fragment(coord, mats);

    // Will use linear equations since this fragment is not uniform
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    const Matrix3x3* lhs = fragEquations.getLHSInverse();
    std::vector<REAL> lhsExpected({ asREAL(5.064935064935065E-12), 0, 0, 0, asREAL(5.064935064935065E-12), 0, 0, 0, asREAL(5.064935064935065E-12) });
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations.getRHS(1);
    std::vector<REAL> rhsExpected({ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations.getRHS(9);
    std::vector<REAL> rhsExpected2({ asREAL(-1.794871794871795e10), asREAL(-1.6826923076923077e10), asREAL(1.682692307692307e9),
                                     asREAL(-1.6826923076923077e10), asREAL(-1.794871794871795e10), asREAL(1.682692307692307e9),
                                     asREAL(-1.682692307692307e9), asREAL(-1.682692307692307e9), asREAL(-1.1217948717948723e9) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));
}

TEST_F(MCEFactoryTests, MixSteelNullVertex_2_0_0) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.EMPTY);
    mats[6] = &Templates::Mat.STEEL;

    ProblemFragment fragment(coord, mats);

    // Will use linear equations since this fragment is not uniform
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    RHSMatricesReader reader;
    std::vector<Matrix3x3> rhsExpectedList = reader.read("test/src/unit/material/data/MCEFactoryTests.MixSteelNullVertex_2_0_0.txt");

    for (int i = 0; i < 27; i++) {
        if (i == 13) {
            const Matrix3x3* lhsActual = fragEquations.getLHSInverse();
            Matrix3x3 lhsExpected = rhsExpectedList.at(i);
            ASSERT_TRUE(closeEqual(*lhsActual, lhsExpected)) << "Expected\n" << lhsExpected << "\ngot\n " << *lhsActual;
        }
        const Matrix3x3* rhsActual = fragEquations.getRHS(i);
        Matrix3x3 rhsExpected = rhsExpectedList.at(i);
        ASSERT_TRUE(closeEqual(*rhsActual, rhsExpected)) << "Expected\n" << rhsExpected << "\ngot\n" << *rhsActual;
    }

}

TEST_F(MCEFactoryTests, MixSteelNullVertex_2_1_1) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.EMPTY);
    mats[0] = mats[2] = mats[4] = mats[6] = &Templates::Mat.STEEL;

    ProblemFragment fragment(coord, mats);

    // Will use linear equations since this fragment is not uniform
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    RHSMatricesReader reader;
    std::vector<Matrix3x3> rhsExpectedList = reader.read("test/src/unit/material/data/MCEFactoryTests.MixSteelNullVertex_2_1_1.txt");

    for (int i = 0; i < 27; i++) {
        if (i == 13) {
            const Matrix3x3* lhsActual = fragEquations.getLHSInverse();
            Matrix3x3 lhsExpected = rhsExpectedList.at(i);
            ASSERT_TRUE(closeEqual(*lhsActual, lhsExpected)) << "Expected\n" << lhsExpected << "\ngot\n " << *lhsActual;
        }
        const Matrix3x3* rhsActual = fragEquations.getRHS(i);
        Matrix3x3 rhsExpected = rhsExpectedList.at(i);
        ASSERT_TRUE(closeEqual(*rhsActual, rhsExpected)) << "Expected\n" << rhsExpected << "\ngot\n" << *rhsActual;
    }

}

TEST_F(MCEFactoryTests, SteelNonUniformVoxels) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 0.25, 0.5);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.STEEL);

    ProblemFragment fragment(coord, mats);
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    const Matrix3x3* lhs = fragEquations.getLHSInverse();
    std::vector<REAL> lhsExpected({ asREAL(4.741641337386017E-12), 0, 0, 0, asREAL(1.826697892271663E-12), 0, 0, 0, asREAL(3.5944700460829494E-12) });
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations.getRHS(1);
    std::vector<REAL> rhsExpected({ asREAL(-2.047275641025641E10), 0, 0, 0, asREAL(-6.6746794871794876E10), asREAL(-3.365384615384615E10), 0, asREAL(-3.365384615384615E10), asREAL(-3.3092948717948715E10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations.getRHS(8);
    std::vector<REAL> rhsExpected2({ asREAL(-6.590544871794872E9), asREAL(-4.206730769230769E9), asREAL(2.1033653846153846E9),
                                     asREAL(-4.206730769230769E9), asREAL(-1.7107371794871794E10), asREAL(8.413461538461538E9),
                                     asREAL(2.1033653846153846E9), asREAL(8.413461538461538E9), asREAL(-8.693910256410255E9) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));

    rhs = fragEquations.getRHS(19);
    std::vector<REAL> rhsExpected3({ asREAL(-2.047275641025641E10), asREAL(0.0), asREAL(0.0),
                                     asREAL(0.0), asREAL(-6.6746794871794876E10), asREAL(3.365384615384615E10),
                                     asREAL(0.0), asREAL(3.365384615384615E10), asREAL(-3.3092948717948715E10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected3)));
}

TEST_F(MCEFactoryTests, SteelNullNonUniformVoxels) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 0.25, 0.5);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.EMPTY);
    mats[4] = mats[5] = mats[6] = mats[7] = &Templates::Mat.STEEL;

    ProblemFragment fragment(coord, mats);
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    const Matrix3x3* lhs = fragEquations.getLHSInverse();
    std::vector<REAL> lhsExpected({ 1 / asREAL(1.0544871794871796e11), 0, 0, 0, 1 / asREAL(2.737179487179487e11), 0, 0, 0, 1 / asREAL(1.391025641025641E11) });
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations.getRHS(1);
    std::vector<REAL> rhsExpected({ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations.getRHS(10);
    std::vector<REAL> rhsExpected2({ asREAL(-2.7483974358974358E10), asREAL(0.0), asREAL(-1.1920928955078125E-7),
                                     asREAL(0.0), asREAL(-1.2003205128205127E11), asREAL(6.730769230769229E9),
                                     asREAL(-1.1920928955078125E-7), asREAL(-6.730769230769229E9), asREAL(-1.907051282051282E10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));

    rhs = fragEquations.getRHS(17);
    std::vector<REAL> rhsExpected3({ asREAL(-9.815705128205128E9), asREAL(-8.413461538461538E9), asREAL(-4.2067307692307675E8),
                                     asREAL(-8.413461538461538E9), asREAL(-3.084935897435897E10), asREAL(-1.682692307692307E9),
                                     asREAL(4.2067307692307675E8), asREAL(1.682692307692307E9), asREAL(-5.6089743589743595E9) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected3)));
}


TEST_F(MCEFactoryTests, SteelUniformVoxelsProblem) {
    DiscreteProblem problem = Templates::Problem::STEEL_2_2_2();
    SolutionInspector sol(problem);

    sol.computeMaterialConfigurationEquations();

    ProblemFragment fragment = problem.extractLocalProblem(ettention::Vec3ui(1, 1, 0));
    MaterialConfigurationEquations* fragEquations = sol.getEquationsForFragment(fragment);

    const Matrix3x3* lhs = fragEquations->getLHSInverse();
    std::vector<REAL> lhsExpected({ asREAL(5.064935064935065E-12), 0, 0, 0, asREAL(5.064935064935065E-12), 0, 0, 0, asREAL(5.064935064935065E-12) });
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations->getRHS(1);
    std::vector<REAL> rhsExpected({ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations->getRHS(10);
    std::vector<REAL> rhsExpected2({ asREAL(2.2435897435897438E10), asREAL(0.0), asREAL(-4.76837158203125E-7),
                                     asREAL(0.0), asREAL(-4.4871794871794876E10), asREAL(6.730769230769229E9),
                                     asREAL(-4.76837158203125E-7), asREAL(-6.730769230769229E9), asREAL(2.2435897435897438E10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));

    rhs = fragEquations->getRHS(17);
    std::vector<REAL> rhsExpected3({ asREAL(-1.794871794871795E10), asREAL(-1.6826923076923077E10), asREAL(-1.682692307692307E9),
                                     asREAL(-1.6826923076923077E10), asREAL(-1.794871794871795E10), asREAL(-1.682692307692307E9),
                                     asREAL(1.682692307692307E9), asREAL(1.682692307692307E9), asREAL(-1.1217948717948723E9) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected3)));
}
