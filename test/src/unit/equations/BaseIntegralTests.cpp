#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "equations/LinearBaseIntegrals.h"
#include "equations/QuadraticBaseIntegrals.h"

class BaseIntegralTests : public ::testing::Test {

public:
    BaseIntegralTests() {}
    ~BaseIntegralTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(BaseIntegralTests, LinearIntegrals) {

    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    EXPECT_EQ(linearIntegrals.value(0, 0, 0, 0), static_cast<REAL>(-1.0 * 1.0 / 1.0 / 36.0));
    EXPECT_EQ(linearIntegrals.value(26, 2, 2, 7), static_cast<REAL>(-1.0 / 1.0 * 1.0 * 1.0 / 36.0));

}

TEST_F(BaseIntegralTests, LinearIntegralsNonUniformVoxels) {

    ettention::Vec3<REAL> voxelSize(1.f, 0.5f, 0.25f);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    EXPECT_EQ(linearIntegrals.value(0, 0, 0, 0), static_cast<REAL>(-0.25 * 0.5 / 1.0 / 36));
    EXPECT_EQ(linearIntegrals.value(26, 2, 2, 7), static_cast<REAL>(-1 / 0.25 * 0.5 * 1.0 / 36));

}

TEST_F(BaseIntegralTests, QuadraticIntegrals) {

    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);

    EXPECT_EQ(quadIntegrals.value(0, 0, 0, 0), static_cast<REAL>(-3703 / 86400 / 1.0 * 1.0 * 1.0));
    EXPECT_EQ(quadIntegrals.value(26, 2, 2, 7), static_cast<REAL>(-3703 / 86400 * 1.0 * 1.0 / 1.0));

}


TEST_F(BaseIntegralTests, QuadraticIntegralsNonUniformVoxels) {

    ettention::Vec3<REAL> voxelSize(1, 0.5f, 0.25f);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);

    EXPECT_EQ(quadIntegrals.value(0, 0, 0, 0), static_cast<REAL>(-3703 / 86400 * 1.0 * 0.5 / 0.25));
    EXPECT_EQ(quadIntegrals.value(26, 2, 2, 7), static_cast<REAL>(-3703 / 86400 * 1.0 * 0.5 / 0.25));

}
