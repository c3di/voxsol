#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include "helpers/ImportanceVolumeInspector.h"
#include "helpers/Templates.h"

class ImportanceVolumeTests: public ::testing::Test {

public:
    ImportanceVolumeTests() {}
    ~ImportanceVolumeTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(ImportanceVolumeTests, VolumeInitTest) {
    DiscreteProblem testProblem = Templates::Problem::STEEL_3_3_3();
    NeumannBoundary someForce(libmmv::Vec3<REAL>(0, 0, 1000));
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(0, 0, 0), someForce);
    ImportanceVolumeInspector testVolume(testProblem);

    ASSERT_EQ(2, testVolume.numberOfLevels()) << "Expected a pyramid with 2 levels for solution size 4x4x4";
    ASSERT_EQ(0, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 0, 1))) << "Expected vertex without Neumann boundary to initially have residual 0";
    ASSERT_EQ(1, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 0, 0))) << "Expected vertex with Neumann boundary to initially have residual 1";
}

TEST_F(ImportanceVolumeTests, SimpleResidualProjection) {
    //Test residual projection from level 0 to level 1. Residuals are always combined toward the smaller coordinate:
    //  Example 1D problem with 5 vertices on the lowest level and arbitrary residual values:
    //
    //                           Residuals
    //  Level 0  .__.__.__.__.   1 2 3 4 5
    //           | /   |  /  |
    //  Level 1  ._____._____.   3 7 5
    //           |    /      |
    //  Level 2  .___________.   10  5  

    //3D problem with 5x5x5 vertices
    DiscreteProblem testProblem = Templates::Problem::STEEL(libmmv::Vec3ui(4, 4, 4));
    NeumannBoundary someForce(libmmv::Vec3<REAL>(0, 0, 1000));

    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(0, 0, 0), someForce);

    ImportanceVolumeInspector testVolume(testProblem);
    testVolume.updateAllLevelsAboveZero();

    ASSERT_EQ(3, testVolume.numberOfLevels()) << "Expected a pyramid with 3 levels for solution size 5x5x5";
    ASSERT_EQ(0, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 0, 1))) << "Expected vertex without Neumann boundary to initially have residual 0";
    ASSERT_EQ(1, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 0, 0))) << "Expected vertex with Neumann boundary to initially have residual 1";
    ASSERT_EQ(1, testVolume.getResidualOnLevel(1, VertexCoordinate(0, 0, 0))) << "Expected vertex on top layer to be given residual 1 from the Neumann boundary on layer 0";
}

TEST_F(ImportanceVolumeTests, ResidualProjectionFromProblemToLevelZero) {
    //3D problem with 5x5x5 vertices 
    DiscreteProblem testProblem = Templates::Problem::STEEL(libmmv::Vec3ui(4, 4, 4));
    NeumannBoundary someForce(libmmv::Vec3<REAL>(0, 0, 1000));

    //On level 0 these should be combined as:
    // Residual of 1 at (0,0,0)
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(0, 0, 0), someForce);

    // Residual of 2 at (1,0,0) and 1 at (2,0,0)
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(2, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(3, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(4, 0, 0), someForce);

    // Residual of 1 at (0,2,1) and 1 at (0,3,3)
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(0, 4, 3), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(1, 4, 4), someForce);

    ImportanceVolumeInspector testVolume(testProblem);

    //See above for expected values
    ASSERT_EQ(3, testVolume.numberOfLevels()) << "Expected a pyramid with 3 levels for solution size 5x5x5";
    ASSERT_EQ(1, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 0, 0)));
    ASSERT_EQ(2, testVolume.getResidualOnLevel(0, VertexCoordinate(1, 0, 0)));
    ASSERT_EQ(1, testVolume.getResidualOnLevel(0, VertexCoordinate(2, 0, 0)));
    ASSERT_EQ(1, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 2, 1)));
    ASSERT_EQ(0, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 3, 3)));
    ASSERT_EQ(0, testVolume.getResidualOnLevel(0, VertexCoordinate(1, 2, 1)));
}

TEST_F(ImportanceVolumeTests, ResidualProjection3Levels) {
    //3D problem with 9x9x9 vertices 
    DiscreteProblem testProblem = Templates::Problem::STEEL(libmmv::Vec3ui(8, 8, 8));
    NeumannBoundary someForce(libmmv::Vec3<REAL>(0, 0, 1000));

    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(0, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(1, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(2, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(3, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(4, 0, 0), someForce);
    testProblem.setNeumannBoundaryAtVertex(libmmv::Vec3ui(5, 0, 0), someForce);

    ImportanceVolumeInspector testVolume(testProblem);
    testVolume.updateAllLevelsAboveZero();

    ASSERT_EQ(4, testVolume.numberOfLevels()) << "Expected a pyramid with 4 levels for solution size 9x9x9";

    // Check that level 0 values were combined toward the smaller coordinate
    ASSERT_EQ(2, testVolume.getResidualOnLevel(0, VertexCoordinate(0, 0, 0)));
    ASSERT_EQ(2, testVolume.getResidualOnLevel(0, VertexCoordinate(1, 0, 0)));
    ASSERT_EQ(2, testVolume.getResidualOnLevel(0, VertexCoordinate(2, 0, 0)));
    ASSERT_EQ(0, testVolume.getResidualOnLevel(0, VertexCoordinate(3, 0, 0)));

    // Check that the level 1 values were combined toward the smaller coordinate
    ASSERT_EQ(4, testVolume.getResidualOnLevel(1, VertexCoordinate(0, 0, 0)));
    ASSERT_EQ(2, testVolume.getResidualOnLevel(1, VertexCoordinate(1, 0, 0)));

    // Check that the level 2 values were combined toward the smaller coordinate
    ASSERT_EQ(6, testVolume.getResidualOnLevel(2, VertexCoordinate(0, 0, 0)));
}

