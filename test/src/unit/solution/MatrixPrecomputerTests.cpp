#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "solution/MatrixPrecomputer.h"
#include "helpers/Templates.h"

class MatrixPrecomputerTests : public ::testing::Test {

public:
    MatrixPrecomputerTests() {}
    ~MatrixPrecomputerTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

// NOTE: Hard coded values come from the stomech prototype solver

TEST_F(MatrixPrecomputerTests, UniformSteel) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    MatrixPrecomputer precomp(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat::STEEL);
    ProblemFragment fragment(coord, mats);

    // Will use quadratic equations since this fragment is made of a uniform material
    FragmentSignature fsig;
    precomp.initializeSignatureForFragment(&fsig, fragment);
    
    const Matrix3x3* lhs = fsig.getLHS();
    std::vector<REAL> lhsExpected({ 1.3478290598290596E12, 0, 0, 0, 1.3478290598290596E12, 0, 0, 0, 1.3478290598290596E12 });
    
    EXPECT_TRUE(*lhs == Matrix3x3(lhsExpected));

    const Matrix3x3* rhs = fsig.getRHS(1);
    std::vector<REAL> rhsExpected({-1.723076923076923E10, 0, 0, 0, -6.50940170940171E10, -9.572649572649567E10, 0, -9.572649572649567E10,-6.509401709401708E10 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected));

    rhs = fsig.getRHS(9);
    std::vector<REAL> rhsExpected2({ -6.50940170940171E10, -9.572649572649571E10, 2.980232238769531E-7, 
                                     -9.572649572649571E10, -6.509401709401709E10, 4.172325134277344E-7, 
                                     -2.980232238769531E-7, -4.172325134277344E-7,-1.7230769230769222E10 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected2));
}

TEST_F(MatrixPrecomputerTests, MixSteelNull) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 1, 1);
    MatrixPrecomputer precomp(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat::EMPTY);
    mats[4] = mats[5] = mats[6] = mats[7] = &Templates::Mat::STEEL;
 
    ProblemFragment fragment(coord, mats);

    // Will use linear equations since this fragment is not uniform
    FragmentSignature fsig;
    precomp.initializeSignatureForFragment(&fsig, fragment);

    const Matrix3x3* lhs = fsig.getLHS();
    std::vector<REAL> lhsExpected({ 1.9743589743589746E11, 0, 0, 0, 1.9743589743589746E11, 0, 0, 0, 1.9743589743589746E11 });
    EXPECT_TRUE(*lhs == Matrix3x3(lhsExpected));

    const Matrix3x3* rhs = fsig.getRHS(1);
    std::vector<REAL> rhsExpected({ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected));

    rhs = fsig.getRHS(9);
    std::vector<REAL> rhsExpected2({ -1.794871794871795E10, -1.6826923076923077E10, 1.682692307692307E9,
        -1.6826923076923077E10, -1.794871794871795E10, 1.682692307692307E9,
        -1.682692307692307E9, -1.682692307692307E9,-1.1217948717948723E9 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected2));
}

TEST_F(MatrixPrecomputerTests, SteelNonUniformVoxels) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 0.25, 0.5);
    MatrixPrecomputer precomp(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat::STEEL);

    ProblemFragment fragment(coord, mats);
    FragmentSignature fsig;
    precomp.initializeSignatureForFragment(&fsig, fragment);

    const Matrix3x3* lhs = fsig.getLHS();
    std::vector<REAL> lhsExpected({ 7.198632478632479E11, 0, 0, 0, 1.868581196581196E12, 0, 0, 0, 9.496068376068375E11 });
    EXPECT_TRUE(*lhs == Matrix3x3(lhsExpected));

    const Matrix3x3* rhs = fsig.getRHS(1);
    std::vector<REAL> rhsExpected({ -3.661538461538461E10, 0, 0, 0, -1.143931623931624E11, -9.572649572649567E10, 0, -9.572649572649567E10, -5.695726495726497E10 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected));

    rhs = fsig.getRHS(9);
    std::vector<REAL> rhsExpected2({ -3.541880341880342E10, -4.7863247863247856E10, 7.450580596923828E-8,
        -4.7863247863247856E10, -1.072136752136752E11, 4.172325134277344E-7,
        -7.450580596923828E-8, -4.172325134277344E-7, -2.5846153846153847E10 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected2));
}

TEST_F(MatrixPrecomputerTests, SteelNullNonUniformVoxels) {
    ettention::Vec3ui coord(2, 2, 2);
    ettention::Vec3<REAL> voxelSize(1, 0.25, 0.5);
    MatrixPrecomputer precomp(voxelSize);

    std::vector<Material*> mats(8, &Templates::Mat::EMPTY);
    mats[4] = mats[5] = mats[6] = mats[7] = &Templates::Mat::STEEL;

    ProblemFragment fragment(coord, mats);
    FragmentSignature fsig;
    precomp.initializeSignatureForFragment(&fsig, fragment);

    const Matrix3x3* lhs = fsig.getLHS();
    std::vector<REAL> lhsExpected({ 1.0544871794871796E11, 0, 0, 0, 2.737179487179487E11, 0, 0, 0, 1.391025641025641E11 });
    EXPECT_TRUE(*lhs == Matrix3x3(lhsExpected));

    const Matrix3x3* rhs = fsig.getRHS(1);
    std::vector<REAL> rhsExpected({ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected));

    rhs = fsig.getRHS(9);
    std::vector<REAL> rhsExpected2({ -9.815705128205128E9, -8.413461538461538E9, 4.2067307692307675E8,
        -8.413461538461538E9, -3.084935897435897E10, 1.682692307692307E9,
        -4.2067307692307675E8, -1.682692307692307E9, -5.6089743589743595E9 });
    EXPECT_TRUE(*rhs == Matrix3x3(rhsExpected2));
}
