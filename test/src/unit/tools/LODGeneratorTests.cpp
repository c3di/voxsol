#include "stdafx.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"
#include "libmmv/math/Vec3.h"
#include "helpers/Templates.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "tools/LODGenerator.h"

class LODGeneratorTests : public ::testing::Test {

public:
    LODGeneratorTests() {}
    ~LODGeneratorTests() {}

    bool closeEqual(const libmmv::Vec3<REAL>& a, const libmmv::Vec3<REAL>& b) {
        for (unsigned int i = 0; i < 3; i++) {
            const testing::internal::FloatingPoint<REAL> lhs(a[i]), rhs(b[i]);
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

