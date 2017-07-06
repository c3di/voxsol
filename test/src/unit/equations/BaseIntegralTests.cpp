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

TEST_F(BaseIntegralTests, LinearIntegralsSinglePrecision) {

    ettention::Vec3f voxelSize(1, 1, 1);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    ASSERT_FLOAT_EQ(linearIntegrals.value(0, 0, 0, 0), Real(-1.0 * 1.0 / 1.0 / 36));
    ASSERT_FLOAT_EQ(linearIntegrals.value(26, 2, 2, 7), Real(-1.f / 1.f * 1.f * 1.f / 36.f));

}

TEST_F(BaseIntegralTests, LinearIntegralsDoublePrecision) {

    ettention::Vec3f voxelSize(1, 1, 1);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    ASSERT_DOUBLE_EQ(linearIntegrals.value(0, 0, 0, 0), -1.0 * 1.0 / 1.0 / 36.0);
    ASSERT_DOUBLE_EQ(linearIntegrals.value(26, 2, 2, 7), -1.f / 1.0 * 1.0 * 1.0 / 36.f);

}

TEST_F(BaseIntegralTests, LinearIntegralsNonUniformVoxels) {

    ettention::Vec3f voxelSize(1.f, 0.5f, 0.25f);
    LinearBaseIntegrals linearIntegrals(voxelSize);

    ASSERT_DOUBLE_EQ(linearIntegrals.value(0, 0, 0, 0), -0.25 * 0.5 / 1.0 / 36);
    ASSERT_DOUBLE_EQ(linearIntegrals.value(26, 2, 2, 7), -1 / 0.25 * 0.5 * 1.0 / 36);

}

TEST_F(BaseIntegralTests, QuadraticIntegralsSinglePrecision) {

    ettention::Vec3f voxelSize(1, 1, 1);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);

    ASSERT_FLOAT_EQ(quadIntegrals.value(0, 0, 0, 0), -3703 / 86400 / 1.f * 1.f * 1.f);
    ASSERT_FLOAT_EQ(quadIntegrals.value(26, 2, 2, 7), -3703 / 86400 * 1.f * 1.f / 1.f);

}

TEST_F(BaseIntegralTests, QuadraticIntegralsDoublePrecision) {

    ettention::Vec3f voxelSize(1, 1, 1);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);

    ASSERT_DOUBLE_EQ(quadIntegrals.value(0, 0, 0, 0), -3703 / 86400 / 1.0 * 1.0 * 1.0);
    ASSERT_DOUBLE_EQ(quadIntegrals.value(26, 2, 2, 7), -3703 / 86400 * 1.0 * 1.0 / 1.0);

}

TEST_F(BaseIntegralTests, QuadraticIntegralsNonUniformVoxels) {

    ettention::Vec3f voxelSize(1, 0.5f, 0.25f);
    QuadraticBaseIntegrals quadIntegrals(voxelSize);

    ASSERT_DOUBLE_EQ(quadIntegrals.value(0, 0, 0, 0), -3703 / 86400 * 1.0 * 0.5 / 0.25);
    ASSERT_DOUBLE_EQ(quadIntegrals.value(26, 2, 2, 7), -3703 / 86400 * 1.0 * 0.5 / 0.25);

}
