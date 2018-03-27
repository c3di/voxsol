#include "stdafx.h"
#include "gtest/gtest.h"
#include "libmmv/math/Vec3.h"
#include "helpers/IOHelper.h"
#include "io/MRCVoxelImporter.h"
#include "material/MaterialDictionary.h"
#include "problem/DiscreteProblem.h"
#include "problem/boundaryconditions/BoundaryProjector.h"

class BoundaryProjectionTests : public ::testing::Test {

public:
    BoundaryProjectionTests() {}
    ~BoundaryProjectionTests() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

};

TEST_F(BoundaryProjectionTests, SimpleDirichletProjection) {
    std::string filePath = IOHelper::getAbsolutePathToFile("test/data/8x8x8_with_4x4x6_cube.mrc");
    MRCVoxelImporter importer(filePath);
    MaterialDictionary matDict = importer.extractMaterialDictionary();
    DiscreteProblem problem(importer.getDimensionsInVoxels(), importer.getVoxelSizeInMeters(), &matDict);
    importer.populateDiscreteProblem(&problem);

    DirichletBoundary dirichlet(DirichletBoundary::FIXED_ALL);
    BoundaryProjector projector(&problem);
    projector.projectDirichletBoundaryAlongNegZ(&dirichlet);

    ASSERT_FALSE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(0, 0, 6)).hasFixedAxes()) << "Expected voxel with empty material to have no dirichlet boundary";
    ASSERT_TRUE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(3, 3, 7)).hasFixedAxes()) << "Expected top z-layer surface vertex to have received a dirichlet boundary";
    ASSERT_FALSE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(3, 3, 6)).hasFixedAxes()) << "Expected vertex underneath top-layer surface to have no dirichlet boundary";
}

TEST_F(BoundaryProjectionTests, DirichletOntoDiamondShape) {
    std::string filePath = IOHelper::getAbsolutePathToFile("test/data/8x8x8_with_diamond.mrc");
    MRCVoxelImporter importer(filePath);
    MaterialDictionary matDict = importer.extractMaterialDictionary();
    DiscreteProblem problem(importer.getDimensionsInVoxels(), importer.getVoxelSizeInMeters(), &matDict);
    importer.populateDiscreteProblem(&problem);

    DirichletBoundary dirichlet(DirichletBoundary::FIXED_ALL);
    BoundaryProjector projector(&problem);
    projector.projectDirichletBoundaryAlongNegZ(&dirichlet);

    // The top of the pyramid is 2x2 centered voxels. The next layer is 3x3 voxels. We're projecting in the -z direction. This means of the 8 vertices for each of the top voxels, all but the inner 
    // vertex on the bottom side should receive a boundary, since the voxels in the 3x3 layer also get a boundary and they share the bottom vertices with the 2x2 layer. 
    ASSERT_TRUE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(3, 3, 8)).hasFixedAxes()) << "Expected topleft vertex of top voxel to be given a dirichlet boundary";
    ASSERT_TRUE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(4, 4, 8)).hasFixedAxes()) << "Expected topright vertex of top voxel to be given a dirichlet boundary";
    ASSERT_FALSE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(4, 4, 7)).hasFixedAxes()) << "Expected bottomright (inner) vertex of top voxel to have no boundary";
    ASSERT_TRUE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(3, 3, 7)).hasFixedAxes()) << "Expected bottomleft vertex of top voxel to be given a dirichlet boundary";

    // at z==4 we've reached the half way point of the diamond, covering the entire layer with non-empty materials. The outermost voxels should also have received a boundary
    // in the raycasting from the -z direction. Note the check is for z==5 since we're changing from voxel to vertex space.
    ASSERT_TRUE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(0, 0, 5)).hasFixedAxes()) << "Expected voxels at z==4 to be given a dirichlet boundary";
    ASSERT_TRUE(problem.getDirichletBoundaryAtVertex(libmmv::Vec3ui(8, 8, 5)).hasFixedAxes()) << "Expected voxels at z==4 to be given a dirichlet boundary";
}

TEST_F(BoundaryProjectionTests, SimpleNeumannProjection) {
    std::string filePath = IOHelper::getAbsolutePathToFile("test/data/8x8x8_with_4x4x6_cube.mrc");
    MRCVoxelImporter importer(filePath);
    MaterialDictionary matDict = importer.extractMaterialDictionary();
    DiscreteProblem problem(importer.getDimensionsInVoxels(), importer.getVoxelSizeInMeters(), &matDict);
    importer.populateDiscreteProblem(&problem);

    libmmv::Vec3<REAL> stress(0, 0, 1000);
    NeumannBoundary neumann(stress);

}
