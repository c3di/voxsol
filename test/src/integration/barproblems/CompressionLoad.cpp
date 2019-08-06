#include "stdafx.h"
#include <unordered_map>
#include <chrono>
#include "gtest/gtest.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "problem/boundaryconditions/BoundaryProjector.h"
#include "material/MaterialDictionary.h"
#include "material/MaterialFactory.h"
#include "integrationHelpers/Templates.h"
#include "io/VTKSolutionWriter.h"
#include "io/VTKSamplingVisualizer.h"
#include "io/VTKImportanceVisualizer.h"
#include "gpu/kernels/SolveDisplacementKernel.h"
#include "gpu/sampling/ImportanceBlockSampler.h"
#include "problem/ProblemInstance.h"
#include "solution/samplers/SequentialBlockSampler.h"
#include "gpu/GPUParameters.h"
#include "gpu/sampling/WaveSampler.h"

bool doOutputTestResultsAsVTK = true;

class CompressionLoadTest : public ::testing::Test {

public:
    CompressionLoadTest() {}
    ~CompressionLoadTest() {}

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

    void fillProblemWithMaterial(DiscreteProblem* problem, Material& material) {
        libmmv::Vec3ui size = problem->getSize();
        for (unsigned int index = 0; index < size.x * size.y * size.z; index++) {
            problem->setMaterial(index, material.id);
        }
    }

};

TEST_F(CompressionLoadTest, SimpleCompression) {
    libmmv::Vec3ui discretization(10, 10, 100);
    libmmv::Vec3<REAL> voxelSize(asREAL(0.1), asREAL(0.1) , asREAL(0.1));
    
    ProblemInstance problemInstance;
    problemInstance.initFromParameters(discretization, voxelSize);
    problemInstance.materialDictionary.addMaterial(Templates::Mat.STEEL);
    fillProblemWithMaterial(problemInstance.getProblemLOD(0), Templates::Mat.STEEL);

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    BoundaryProjector bProjector(problemInstance.getProblemLOD(0), ProblemSide::POSITIVE_Z);
    bProjector.setMaxProjectionDepth(2);
    bProjector.projectDirichletBoundary(&fixed);

    bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Z);
    REAL totalNeumannForceNewtons = asREAL(-1e9);
    bProjector.projectNeumannBoundary(totalNeumannForceNewtons);

    problemInstance.createAdditionalLODs(2);

    for (int i = 2; i >= 0; i--) {
        SequentialBlockSampler sampler(problemInstance.getSolutionLOD(i), BLOCK_SIZE);

        int totalSteps = problemInstance.solveLOD(i, asREAL(1e-7), &sampler);
        if (i > 0) {
            problemInstance.projectCoarseSolutionToFinerSolution(i, i-1);
        }

    }

    Solution* sol = problemInstance.getSolutionLOD(0);
    std::vector<Vertex>* vertices = sol->getVertices();

    REAL maxDisplacement = 0;

    for (auto it = vertices->begin(); it != vertices->end(); it++) {
        if (abs(it->z) > maxDisplacement) {
            maxDisplacement = it->z;
        }
    }

    // Analytical solution is max_disp = F*L / E*A for F=1e9, L=10, E=210e9, A=1
    EXPECT_NEAR(asREAL(-0.047419), maxDisplacement, 0.001);

    if (doOutputTestResultsAsVTK) {
        VTKSolutionWriter vis(problemInstance.getSolutionLOD(0));
        vis.writeEntireStructureToFile("d:\\tmp\\integration_SimpleCompression.vtk");
    }
   
}


TEST_F(CompressionLoadTest, SimpleCompressionAnisotropicVoxels) {
    libmmv::Vec3ui discretization(10, 5, 100);
    libmmv::Vec3<REAL> voxelSize(asREAL(0.1), asREAL(0.2), asREAL(0.1));

    ProblemInstance problemInstance;
    problemInstance.initFromParameters(discretization, voxelSize);
    problemInstance.materialDictionary.addMaterial(Templates::Mat.STEEL);
    fillProblemWithMaterial(problemInstance.getProblemLOD(0), Templates::Mat.STEEL);

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    BoundaryProjector bProjector(problemInstance.getProblemLOD(0), ProblemSide::POSITIVE_Z);
    bProjector.setMaxProjectionDepth(2);
    bProjector.projectDirichletBoundary(&fixed);

    bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Z);
    REAL totalNeumannForceNewtons = asREAL(-1e9);
    bProjector.projectNeumannBoundary(totalNeumannForceNewtons);

    problemInstance.createAdditionalLODs(0);

    for (int i = 0; i >= 0; i--) {
        SequentialBlockSampler sampler(problemInstance.getSolutionLOD(i), BLOCK_SIZE);

        int totalSteps = problemInstance.solveLOD(i, asREAL(1e-7), &sampler);
        if (i > 0) {
            problemInstance.projectCoarseSolutionToFinerSolution(i, i - 1);
        }
    }

    Solution* sol = problemInstance.getSolutionLOD(0);
    std::vector<Vertex>* vertices = sol->getVertices();

    REAL maxDisplacement = 0;

    for (auto it = vertices->begin(); it != vertices->end(); it++) {
        if (abs(it->z) > maxDisplacement) {
            maxDisplacement = it->z;
        }
    }

    // Analytical solution is max_disp = F*L / E*A for F=1e9, L=10, E=210e9, A=1
    EXPECT_NEAR(asREAL(-0.047419), maxDisplacement, 0.001);

    if (doOutputTestResultsAsVTK) {
        VTKSolutionWriter vis(problemInstance.getSolutionLOD(0));
        vis.writeEntireStructureToFile("d:\\tmp\\integration_SimpleCompressionAnisotropicVoxels.vtk");
    }

}

TEST_F(CompressionLoadTest, SimpleCompressionNonUnitArea) {
    libmmv::Vec3ui discretization(100, 10, 10); // uniform voxel size 0.1 with 100,15,10 produces the same solution
    libmmv::Vec3<REAL> voxelSize(asREAL(0.1), asREAL(0.15), asREAL(0.1));

    ProblemInstance problemInstance;
    problemInstance.initFromParameters(discretization, voxelSize);
    problemInstance.materialDictionary.addMaterial(Templates::Mat.STEEL);
    fillProblemWithMaterial(problemInstance.getProblemLOD(0), Templates::Mat.STEEL);

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    BoundaryProjector bProjector(problemInstance.getProblemLOD(0), ProblemSide::POSITIVE_Z);
    bProjector.setMaxProjectionDepth(2);
    bProjector.projectDirichletBoundary(&fixed);

    bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Z);
    REAL totalNeumannForceNewtons = asREAL(-1e9);
    bProjector.projectNeumannBoundary(totalNeumannForceNewtons);

    problemInstance.createAdditionalLODs(1);

    for (int i = 1; i >= 0; i--) {
        SequentialBlockSampler sampler(problemInstance.getSolutionLOD(i), BLOCK_SIZE);

        int totalSteps = problemInstance.solveLOD(i, asREAL(1e-7), &sampler);
        if (i > 0) {
            problemInstance.projectCoarseSolutionToFinerSolution(i, i - 1);
        }
    }

    Solution* sol = problemInstance.getSolutionLOD(0);
    std::vector<Vertex>* vertices = sol->getVertices();

    REAL maxDisplacement = 0;

    for (auto it = vertices->begin(); it != vertices->end(); it++) {
        if (abs(it->x) > maxDisplacement) {
            maxDisplacement = it->x;
        }
    }

    // Analytical solution is max_disp = F*L / E*A for F=1e9, L=10, E=210e9, A=1.5
    EXPECT_NEAR(asREAL(-0.031746), maxDisplacement, 0.001);

    if (doOutputTestResultsAsVTK) {
        VTKSolutionWriter vis(problemInstance.getSolutionLOD(0));
        vis.writeEntireStructureToFile("d:\\tmp\\integration_SimpleCompressionNonUnitArea.vtk");
    }

}

TEST_F(CompressionLoadTest, SimpleCompressionAnisotropicUnitArea) {
    libmmv::Vec3ui discretization(100, 10, 20); 
    libmmv::Vec3<REAL> voxelSize(asREAL(0.1), asREAL(0.1), asREAL(0.05));

    ProblemInstance problemInstance;
    problemInstance.initFromParameters(discretization, voxelSize);
    problemInstance.materialDictionary.addMaterial(Templates::Mat.STEEL);
    fillProblemWithMaterial(problemInstance.getProblemLOD(0), Templates::Mat.STEEL);

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    BoundaryProjector bProjector(problemInstance.getProblemLOD(0), ProblemSide::POSITIVE_X);
    bProjector.setMaxProjectionDepth(2);
    bProjector.projectDirichletBoundary(&fixed);

    bProjector.setProjectionDirection(ProblemSide::NEGATIVE_X);
    REAL totalNeumannForceNewtons = asREAL(-1e9);
    bProjector.projectNeumannBoundary(totalNeumannForceNewtons);

    problemInstance.createAdditionalLODs(1);

    for (int i = 1; i >= 0; i--) {
        SequentialBlockSampler sampler(problemInstance.getSolutionLOD(i), BLOCK_SIZE);

        int totalSteps = problemInstance.solveLOD(i, asREAL(1e-7), &sampler);
        if (i > 0) {
            problemInstance.projectCoarseSolutionToFinerSolution(i, i - 1);
        }
    }

    Solution* sol = problemInstance.getSolutionLOD(0);
    std::vector<Vertex>* vertices = sol->getVertices();

    REAL maxDisplacement = 0;

    for (auto it = vertices->begin(); it != vertices->end(); it++) {
        if (abs(it->x) > maxDisplacement) {
            maxDisplacement = it->x;
        }
    }

    // Analytical solution is max_disp = F*L / E*A for F=1e9, L=10, E=210e9, A=1
    EXPECT_NEAR(asREAL(-0.047419), maxDisplacement, 0.001);

    if (doOutputTestResultsAsVTK) {
        VTKSolutionWriter vis(problemInstance.getSolutionLOD(0));
        vis.writeEntireStructureToFile("d:\\tmp\\integration_SimpleCompressionNonUnitArea.vtk");
    }

}
