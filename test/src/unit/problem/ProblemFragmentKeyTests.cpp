#include "stdafx.h"
#include <unordered_map>
#include "gtest/gtest.h"
#include "problem/ProblemFragment.h"


class ProblemFragmentKeyTests : public ::testing::Test {

public:
    ProblemFragmentKeyTests() {}
    ~ProblemFragmentKeyTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(ProblemFragmentKeyTests, EmptyInitialization) {
    
    ProblemFragment frag(ettention::Vec3ui(0, 0, 0));
    ProblemFragmentKey fragKey = frag.key();
    Material mat(7830, 210e9, 0.3, 0.0, 1);

    unsigned char ids = 0;
    for (int i = 0; i < 8; i++) {
        ids += fragKey[i];
    }

    // Key should only contain the empty material with id 0
    ASSERT_EQ(ids, 0);
    
    frag.setMaterial(0, &mat);
    fragKey = frag.key();
    ids = 0;
    for (int i = 0; i < 8; i++) {
        ids += fragKey[i];
    }

    // Key now contains one material with id=1, so the whole key should be 1
    ASSERT_EQ(ids, 1);

}

TEST_F(ProblemFragmentKeyTests, KeyHash) {

    std::unordered_map<ProblemFragmentKey, unsigned int> map;
    ProblemFragment frag(ettention::Vec3ui(0, 0, 0));
    Material mat(7830, 210e9, 0.3, 0.0, 1);
    frag.setMaterial(0, &mat);

    ProblemFragmentKey fragKey = frag.key();
    map[fragKey] = 5;

    ASSERT_EQ(map.count(fragKey), 1);

    frag.setMaterial(1, &mat);
    ProblemFragmentKey fragKey2 = frag.key();
    map[fragKey2] = 6;

    ASSERT_TRUE(map.count(fragKey) == 1 && map.count(fragKey2) == 1);
    ASSERT_TRUE(map[fragKey] == 5 && map[fragKey2] == 6);

}

TEST_F(ProblemFragmentKeyTests, Equality) {

    ProblemFragment fragOne(ettention::Vec3ui(0, 0, 0));
    ProblemFragment fragTwo(ettention::Vec3ui(0, 0, 0));
    Material matA(7830, 210e9, 0.3, 0.0, 1);
    Material matB(8830, 220e9, 0.4, 0.0, 2);
    
    fragOne.setMaterial(0, &matA);
    fragOne.setMaterial(4, &matA);
    fragTwo.setMaterial(0, &matA);
    fragTwo.setMaterial(4, &matA);

    ProblemFragmentKey keyOne = fragOne.key();
    ProblemFragmentKey keyTwo = fragTwo.key();

    ASSERT_EQ(keyOne, keyTwo);

    fragTwo.setMaterial(0, &matB);
    keyTwo = fragTwo.key();

    ASSERT_NE(keyOne, keyTwo);

    fragOne.setMaterial(0, &matB);
    keyOne = fragOne.key();

    ASSERT_EQ(keyOne, keyTwo);
}
