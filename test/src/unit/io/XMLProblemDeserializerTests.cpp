#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "material/MaterialDictionary.h"
#include "helpers/Templates.h"
#include "io/XMLProblemDeserializer.h"
#include "helpers/IOHelper.h"

class XMLProblemDeserializerTests : public ::testing::Test {

public:
    XMLProblemDeserializerTests() {}
    ~XMLProblemDeserializerTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};


TEST_F(XMLProblemDeserializerTests, 8x8x8_with_4x4x6_cube_xml) {
    //NOTE: The problem size is 8x8x8, but the cube itself is 6x6x6 aluminium surrounded by a border of empty voxels
    std::string filePath = IOHelper::getAbsolutePathToFile("test/data/8x8x8_with_4x4x6_cube.xml");
    XMLProblemDeserializer xmlDeserializer(filePath);

    ProblemInstance problemInstance = xmlDeserializer.getProblemInstance();
    ASSERT_EQ(problemInstance.getNumberOfLODs(), 2) << "Expected 2 LODs total"; //includes full res also (LOD 0)
    DiscreteProblem* problemlod0 = problemInstance.getProblemLOD(0);
    libmmv::Vec3ui expectedSize(8, 8, 8);
    libmmv::Vec3<REAL> expectedVoxelSize(asREAL(0.125), asREAL(0.125), asREAL(0.125));

    ASSERT_EQ(problemlod0->getSize(), expectedSize) << "Expected problem size to be 8x8x8 voxels";
    ASSERT_EQ(problemlod0->getVoxelSize(), expectedVoxelSize) << "Expected voxel size to be 1/8";

    MaterialDictionary matDict = problemInstance.materialDictionary;
    ASSERT_TRUE(matDict.contains(0) && matDict.contains(6)) << "Expected two materials, empty (id=0) and aluminium (id=6)";

    Material* aluminium = matDict.getMaterialById(6);
    ASSERT_EQ(aluminium->poissonRatio, asREAL(0.35)) << "Expected aluminium material to have poisson ratio 0.35";
    ASSERT_EQ(aluminium->youngsModulus, asREAL(70000000000)) << "Expected aluminium material to have young's modulus of 7e10 Pa";

    // Boundaries were applied to 3 sides of the cube which has size 5x5x7 in vertex space.
    // In -X: 5x7 = 35 vertices
    // In -Y: 5x7 - 7 = 28 vertices (shared edge with -X)
    // In +Z: 5x5 - 5 - 4 = 16 vertices (shared edges with -X and -Y)
    // Total: 79 vertices with a Dirichlet boundary
    size_t numDirichletBoundaryVertices = problemlod0->getDirichletBoundaryMap()->size();
    ASSERT_EQ(numDirichletBoundaryVertices, 79) << "Expected a total of 79 vertices to be assigned a Dirichlet boundary";

    // Neumann boundary was applied to -Z, 5x5 = 25 vertices total
    size_t numNeumannBoundaryVertices = problemlod0->getNeumannBoundaryMap()->size();
    ASSERT_EQ(numNeumannBoundaryVertices, 25) << "Expected 5x5=25 vertices to be assigned a Neumann boundary";
}
