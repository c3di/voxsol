#include "stdafx.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"
#include "libmmv/math/Vec3.h"
#include "material/MaterialConfigurationEquationsFactory.h"
#include "material/MaterialConfigurationEquations.h"
#include "helpers/Templates.h"

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

TEST_F(MCEFactoryTests, UniformSteel) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.STEEL);
    ProblemFragment fragment(coord, mats);

    // Will use quadratic equations since this fragment is made of a uniform material
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    const Matrix3x3* lhs = fragEquations.getLHSInverse();
    std::vector<REAL> lhsExpected({ 1 / asREAL(1.3478290598290596e12), 0, 0, 0, 1 / asREAL(1.3478290598290596e12), 0, 0, 0, 1 / asREAL(1.3478290598290596e12) });
    
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations.getRHS(1);
    std::vector<REAL> rhsExpected({asREAL(-1.723076923076923e10), 0, 0, 0, asREAL(-6.50940170940171e10), asREAL(-9.572649572649567e10), 0, asREAL(-9.572649572649567e10),asREAL(-6.509401709401708e10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations.getRHS(9);
    std::vector<REAL> rhsExpected2({ asREAL(-6.50940170940171e10), asREAL(-9.572649572649571e10), asREAL(2.980232238769531e-7),
                                     asREAL(-9.572649572649571e10), asREAL(-6.509401709401709e10), asREAL(4.172325134277344e-7),
                                     asREAL(-2.980232238769531e-7), asREAL(-4.172325134277344e-7), asREAL(-1.7230769230769222e10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));
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
    std::vector<REAL> lhsExpected({ 1 / asREAL(1.9743589743589746e11), 0, 0, 0, 1 / asREAL(1.9743589743589746e11), 0, 0, 0, 1 / asREAL(1.9743589743589746e11) });
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

TEST_F(MCEFactoryTests, SteelNonUniformVoxels) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 0.25, 0.5);
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat.STEEL);

    ProblemFragment fragment(coord, mats);
    MaterialConfigurationEquations fragEquations;
    mceFactory.initializeEquationsForFragment(&fragEquations, fragment);

    const Matrix3x3* lhs = fragEquations.getLHSInverse();
    std::vector<REAL> lhsExpected({ 1 / asREAL(7.198632478632479e11), 0, 0, 0, 1 / asREAL(1.868581196581196e12), 0, 0, 0, 1 / asREAL(9.496068376068375e11) });
    EXPECT_TRUE(closeEqual(*lhs, Matrix3x3(lhsExpected)));

    const Matrix3x3* rhs = fragEquations.getRHS(1);
    std::vector<REAL> rhsExpected({ asREAL(-3.661538461538461e10), 0, 0, 0, asREAL(-1.143931623931624e11), asREAL(-9.572649572649567e10), 0, asREAL(-9.572649572649567e10), asREAL(-5.695726495726497e10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected)));

    rhs = fragEquations.getRHS(9);
    std::vector<REAL> rhsExpected2({ asREAL(-3.541880341880342e10), asREAL(-4.7863247863247856e10), asREAL(7.450580596923828e-8),
                                     asREAL(-4.7863247863247856e10), asREAL(-1.072136752136752e11), asREAL(4.172325134277344e-7),
                                     asREAL(-7.450580596923828e-8), asREAL(-4.172325134277344e-7), asREAL(-2.5846153846153847e10) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));
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

    rhs = fragEquations.getRHS(9);
    std::vector<REAL> rhsExpected2({ asREAL(-9.815705128205128e9), asREAL(-8.413461538461538e9), asREAL(4.2067307692307675e8),
                                     asREAL(-8.413461538461538e9), asREAL(-3.084935897435897e10), asREAL(1.682692307692307e9),
                                     asREAL(-4.2067307692307675e8), asREAL(-1.682692307692307e9), asREAL(-5.6089743589743595e9) });
    EXPECT_TRUE(closeEqual(*rhs, Matrix3x3(rhsExpected2)));
}
