#include "stdafx.h"
#include <unordered_map>
#include <chrono>
#include "gtest/gtest.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "problem/boundaryconditions/BoundaryProjector.h"
#include "problem/ProblemInstance.h"
#include "helpers/Templates.h"
#include "solution/samplers/SequentialBlockSampler.h"
#include "gpu/GPUParameters.h"
#include "io/VTKSolutionWriter.h"

class InitialDisplacementTest : public ::testing::Test {

public:
    InitialDisplacementTest() {}
    ~InitialDisplacementTest() {}

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

    bool doOutputTestResultsAsVTK = false;
};


TEST_F(InitialDisplacementTest, SimpleInitialDisplacement) {
    libmmv::Vec3ui discretization(10, 10, 10);
    libmmv::Vec3<REAL> voxelSize(asREAL(0.1), asREAL(0.1), asREAL(0.1));

    ProblemInstance problemInstance;
    problemInstance.initFromParameters(discretization, voxelSize);
    problemInstance.materialDictionary.addMaterial(Templates::Mat.STEEL);
    fillProblemWithMaterial(problemInstance.getProblemLOD(0), Templates::Mat.STEEL);

    problemInstance.createAdditionalLODs(0);

    for (int i = 0; i <= 0; i++) {
        DisplacementBoundary initialDisplacement(libmmv::Vec3<REAL>(0, 0, asREAL(0.01))); //1% of problem size

        BoundaryProjector bProjector(problemInstance.getProblemLOD(0), ProblemSide::NEGATIVE_Z);
        bProjector.setMaxProjectionDepth(2);
        bProjector.projectDisplacementBoundary(&initialDisplacement);

        bProjector.setProjectionDirection(ProblemSide::POSITIVE_Z);
        DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
        bProjector.projectDirichletBoundary(&fixed);
    }

    problemInstance.computeMaterialConfigurationEquations();

    for (int i = 0; i >= 0; i--) {
        SequentialBlockSampler sampler(problemInstance.getSolutionLOD(i), BLOCK_SIZE);

        int totalSteps = problemInstance.solveLOD(i, asREAL(1e-6), &sampler);
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
    EXPECT_NEAR(asREAL(0.01), maxDisplacement, 0.001);

    if (doOutputTestResultsAsVTK) {
        VTKSolutionWriter vis(problemInstance.getSolutionLOD(0));
        vis.writeEntireStructureToFile("integration_SimpleInitialDisplacement.vtk");
    }

}

