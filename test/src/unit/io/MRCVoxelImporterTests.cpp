#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "material/MaterialDictionary.h"
#include "helpers/Templates.h"
#include "io/MRCVoxelImporter.h"
#include "helpers/IOHelper.h"

class MRCVoxelImporterTests : public ::testing::Test {

public:
    MRCVoxelImporterTests() {}
    ~MRCVoxelImporterTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(MRCVoxelImporterTests, SimpleExample) {
    std::string filePath = IOHelper::getAbsolutePathToFile("test/data/8x8x8_with_4x4x6_cube.mrc");
    MRCVoxelImporter importer(filePath);
    MaterialDictionary matDict = importer.extractMaterialDictionary();
    DiscreteProblem problem = importer.extractDiscreteProblem(&matDict);

    ASSERT_EQ(problem.getSize(), libmmv::Vec3ui(8, 8, 8)) << "Expected problem size to be 8x8x8";
    ASSERT_EQ(problem.getMaterial(libmmv::Vec3ui(0, 0, 0))->id, 0) << "Expected first voxel to have empty material with id 0";
    ASSERT_EQ(problem.getMaterial(libmmv::Vec3ui(4, 4, 0))->id, 0) << "Expected voxel at 4,4,0 to have empty material";
    ASSERT_NE(problem.getMaterial(libmmv::Vec3ui(4, 4, 1))->id, 0) << "Expected voxel at 4,4,1 to have non-empty material";
}
