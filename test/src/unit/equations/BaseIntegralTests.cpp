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

    libmmv::Vec3<REAL> voxelSize(1, 1, 1);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    EXPECT_EQ(linearIntegrals.value(0, 0, 0, 0), asREAL(-1.0 * 1.0 / 1.0 / 36.0));
    EXPECT_EQ(linearIntegrals.value(26, 2, 2, 7), asREAL(-1.0 / 1.0 * 1.0 * 1.0 / 36.0));

}

TEST_F(BaseIntegralTests, LinearIntegralsNonUniformVoxels) {

    libmmv::Vec3<REAL> voxelSize(1.f, 0.5f, 0.25f);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    EXPECT_EQ(linearIntegrals.value(0, 0, 0, 0), asREAL(-0.25 * 0.5 / 1.0 / 36));
    EXPECT_EQ(linearIntegrals.value(26, 2, 2, 7), asREAL(-1 / 0.25 * 0.5 * 1.0 / 36));

}

TEST_F(BaseIntegralTests, QuadraticIntegrals) {

    libmmv::Vec3<REAL> voxelSize(1, 1, 1);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);

    EXPECT_EQ(quadIntegrals.value(0, 0, 0, 0), asREAL(-3703) / 86400 / 1.0 * 1.0 * 1.0);
    EXPECT_EQ(quadIntegrals.value(26, 2, 2, 7), asREAL(-3703) / 86400 * 1.0 * 1.0 / 1.0);

}


TEST_F(BaseIntegralTests, QuadraticIntegralsNonUniformVoxels) {

    libmmv::Vec3<REAL> voxelSize(1, 0.5f, 0.25f);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);
    EXPECT_EQ(quadIntegrals.value(0, 0, 0, 0), asREAL(-3703) / 86400 / 1.0 * 0.5 * 0.25);
    EXPECT_EQ(quadIntegrals.value(26, 2, 2, 7), asREAL(-3703) / 86400 * 1.0 * 0.5 / 0.25);

}
