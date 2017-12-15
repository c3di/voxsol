#include "stdafx.h"
#include <unordered_map>
#include "gtest/gtest.h"
#include "problem/ProblemFragment.h"
#include "material/MaterialConfiguration.h"


class MaterialConfigurationTests : public ::testing::Test {

public:
    MaterialConfigurationTests() {}
    ~MaterialConfigurationTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(MaterialConfigurationTests, EmptyInitialization) {
    
    ProblemFragment frag(libmmv::Vec3ui(0, 0, 0));
    MaterialConfiguration fragKey = frag.getMaterialConfiguration();
    Material mat(asREAL(210e9), asREAL(0.3), 1);

    unsigned char ids = 0;
    for (int i = 0; i < 8; i++) {
        ids += fragKey[i];
    }

    // Key should only contain the empty material with id 0
    ASSERT_EQ(ids, 0);
    
    frag.setMaterial(0, &mat);
    fragKey = frag.getMaterialConfiguration();
    ids = 0;
    for (int i = 0; i < 8; i++) {
        ids += fragKey[i];
    }

    // Key now contains one material with id=1, so the whole key should be 1
    ASSERT_EQ(ids, 1);

}

TEST_F(MaterialConfigurationTests, KeyHash) {

    std::unordered_map<MaterialConfiguration, unsigned int> map;
    ProblemFragment frag(libmmv::Vec3ui(0, 0, 0));
    Material mat(asREAL(210e9), asREAL(0.3), 1);
    frag.setMaterial(0, &mat);

    MaterialConfiguration fragKey = frag.getMaterialConfiguration();
    map[fragKey] = 5;

    ASSERT_EQ(map.count(fragKey), 1);

    frag.setMaterial(1, &mat);
    MaterialConfiguration fragKey2 = frag.getMaterialConfiguration();
    map[fragKey2] = 6;

    ASSERT_TRUE(map.count(fragKey) == 1 && map.count(fragKey2) == 1);
    ASSERT_TRUE(map[fragKey] == 5 && map[fragKey2] == 6);

}

TEST_F(MaterialConfigurationTests, Equality) {

    ProblemFragment fragOne(libmmv::Vec3ui(0, 0, 0));
    ProblemFragment fragTwo(libmmv::Vec3ui(0, 0, 0));
    Material matA(asREAL(210e9), asREAL(0.3), 1);
    Material matB(asREAL(220e9), asREAL(0.4), 2);
    
    fragOne.setMaterial(0, &matA);
    fragOne.setMaterial(4, &matA);
    fragTwo.setMaterial(0, &matA);
    fragTwo.setMaterial(4, &matA);

    MaterialConfiguration keyOne = fragOne.getMaterialConfiguration();
    MaterialConfiguration keyTwo = fragTwo.getMaterialConfiguration();

    ASSERT_EQ(keyOne, keyTwo);

    fragTwo.setMaterial(0, &matB);
    keyTwo = fragTwo.getMaterialConfiguration();

    ASSERT_NE(keyOne, keyTwo);

    fragOne.setMaterial(0, &matB);
    keyOne = fragOne.getMaterialConfiguration();

    ASSERT_EQ(keyOne, keyTwo);
}
