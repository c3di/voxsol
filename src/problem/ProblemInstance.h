#pragma once
#include <vector>
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "gpu/sampling/ResidualVolume.h"
#include "material/MaterialFactory.h"
#include "material/MaterialDictionary.h"
#include "solution/samplers/BlockSampler.h"


class ProblemInstance {
public:

    ProblemInstance();
    ~ProblemInstance();

    MaterialDictionary materialDictionary;

    void initFromMRCStack(std::string& path, bool isStumpMRC = false);
    void initFromMaterialProbeMRC(std::string& path);
    void initFromParameters(libmmv::Vec3ui& discretization, libmmv::Vec3<REAL>& voxelSize);
    void createAdditionalLODs(int numberOfAdditionalLODs);
    void computeMaterialConfigurationEquations();
    void createLODTree(int smallestPermissableDimension);
    void projectCoarseSolutionToFinerSolution(int coarseLod, int fineLod);

    int getNumberOfLODs();
    int solveLOD(int lod, REAL convergenceCriteria, BlockSampler* sampler);

    DiscreteProblem* getProblemLOD(int lod);
    Solution* getSolutionLOD(int lod);
    ResidualVolume* getResidualVolumeLOD(int lod);

    void computeMaterialConfigurationEquationsForLOD(int lod);

    void debug_outputTotalForceForEachLOD();

protected:
    MaterialFactory materialFactory;
    std::vector<DiscreteProblem*> problemLODs;
    std::vector<Solution*> solutionLODs;
    std::vector<ResidualVolume*> residualLODs;
    REAL targetResidualDeltaForConvergence = asREAL(1e-11);

};
