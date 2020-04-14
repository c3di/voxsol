#include "stdafx.h"
#include <iomanip>
#include "ProblemInstance.h"
#include "io/MRCImporter.h"
#include "tools/LODGenerator.h"
#include "gpu/kernels/SolveDisplacementKernel.h"
#include "gpu/sampling/ImportanceBlockSampler.h"
#include "solution/samplers/SequentialBlockSampler.h"
#include "io/VTKImportanceVisualizer.h"
#include "io/VTKSolutionWriter.h"

ProblemInstance::ProblemInstance() 
{
}

ProblemInstance::~ProblemInstance() {
    for (auto it = residualLODs.begin(); it != residualLODs.end(); it++) {
        delete *it;
        *it = nullptr;
    }
    for (auto it = solutionLODs.begin(); it != solutionLODs.end(); it++) {
        delete *it;
        *it = nullptr;
    }
    for (auto it = problemLODs.begin(); it != problemLODs.end(); it++) {
        delete *it;
        *it = nullptr;
    }
}

void ProblemInstance::initFromParameters(libmmv::Vec3ui& discretization, libmmv::Vec3<REAL>& voxelSize) {
    DiscreteProblem* problem = new DiscreteProblem(discretization, voxelSize, &materialDictionary);
    problemLODs.push_back(problem);
    
    Solution* solution = new Solution(problemLODs[0]);
    solutionLODs.push_back(solution);

    ResidualVolume* residual = new ResidualVolume(problemLODs[0]);
    residualLODs.push_back(residual);
}

int ProblemInstance::getNumberOfLODs() {
    return (int)problemLODs.size();
}
DiscreteProblem* ProblemInstance::getProblemLOD(int lod)
{
    if (problemLODs.size() <= lod) {
        throw "Attempted to index an lod that does not exist";
    }
    return problemLODs.at(lod);
}

Solution* ProblemInstance::getSolutionLOD(int lod)
{
    if (solutionLODs.size() <= lod) {
        throw "Attempted to index an lod that does not exist";
    }
    return solutionLODs.at(lod);
}

ResidualVolume* ProblemInstance::getResidualVolumeLOD(int lod)
{
    if (residualLODs.size() <= lod) {
        throw "Attempted to index an lod that does not exist";
    }
    return residualLODs.at(lod);
}

void ProblemInstance::createAdditionalLODs(int numberOfAdditionalLODs) {
    //TODO: config equations need to be computed for each LOD before a lower LOD can be generated from it, this is kind of restricting because 
    // it's (potentially) a very long operation that may need to be done on GPU
    getResidualVolumeLOD(0)->initializePyramidFromProblem();

    LODGenerator lodGen;
    libmmv::Vec3ui size = problemLODs[0]->getSize();
    libmmv::Vec3<REAL> voxelSize = problemLODs[0]->getVoxelSize();

    libmmv::Vec3<REAL> baseDimensions(size.x * voxelSize.x, size.y * voxelSize.y, size.z * voxelSize.z);

    for (int fine = 0; fine < numberOfAdditionalLODs; fine++) {
        int coarse = fine + 1;

        if (size.x > 1) {
            size.x /= 2;
            voxelSize.x = baseDimensions.x / size.x;
        }
        if (size.y > 1) {
            size.y /= 2;
            voxelSize.y = baseDimensions.y / size.y;
        }
        if (size.z > 1) {
            size.z /= 2;
            voxelSize.z = baseDimensions.z / size.z;
        }

        DiscreteProblem* coarseProblem = new DiscreteProblem(size, voxelSize, problemLODs[fine]->getMaterialDictionary());
        lodGen.populateCoarserLevelProblem(coarseProblem, getProblemLOD(fine));
        problemLODs.push_back(coarseProblem);

        Solution* coarseSolution = new Solution(getProblemLOD(coarse));
        lodGen.populateCoarserLevelSolution(coarseSolution, getProblemLOD(coarse), getSolutionLOD(fine));
        solutionLODs.push_back(coarseSolution);

        ResidualVolume* coarseResiduals = new ResidualVolume(getProblemLOD(coarse));
        coarseResiduals->initializePyramidFromProblem();
        residualLODs.push_back(coarseResiduals);
    }
}

void ProblemInstance::computeMaterialConfigurationEquations() {
    for (int i = 0; i < problemLODs.size(); i++) {
        computeMaterialConfigurationEquationsForLOD(i);
    }
}

// Creates as many LODs as necessary to reach the given smallest dimension problem size
void ProblemInstance::createLODTree(int smallestPermissableDimension) {
    libmmv::Vec3ui size(problemLODs[0]->getSize());
    int numAdditionalLODs = 0;
#pragma warning(suppress: 4018) //suppress signed/unsigned mismatch warning
    while (size.x > smallestPermissableDimension && size.y > smallestPermissableDimension && size.z > smallestPermissableDimension) {
        numAdditionalLODs++;
        size = size / 2;
    }
    numAdditionalLODs = std::max(numAdditionalLODs - 1, 0);
    createAdditionalLODs(numAdditionalLODs);
}

void ProblemInstance::projectCoarseSolutionToFinerSolution(int coarseLod, int fineLod) {
    if (coarseLod - fineLod != 1) {
        throw std::exception("Can only project a coarse solution to a neighboring LOD (eg. LOD 2 -> LOD 1)");
    }
    LODGenerator lodGen;
    lodGen.projectDisplacementsToFinerLevel(solutionLODs[coarseLod], solutionLODs[fineLod]);
}

int ProblemInstance::solveLOD(int lod, REAL convergenceCriteria, BlockSampler* sampler) {
    std::cout << "Solving LOD " << lod << " with target residual " << convergenceCriteria << "..." << std::endl;

    ResidualVolume* residualVolume = getResidualVolumeLOD(lod);
    SolveDisplacementKernel kernel(getSolutionLOD(lod), sampler, residualVolume);
    kernel.setNumLaunchesBeforeResidualUpdate(499);
    
    REAL currentResidualError = 1000000;
    int numVerticesNotConverged = 0;
    int totalSteps = 0;
    do {
        kernel.launch();

        totalSteps++;

        if (totalSteps % 500 == 0 || currentResidualError <= convergenceCriteria) {
            currentResidualError = residualVolume->getResidualDeltaToLastUpdate(&numVerticesNotConverged);
            std::cout << "\rCurrent residual: " << currentResidualError << " Steps: "<< totalSteps << "                                         ";
        }
        if (isnan(currentResidualError)) {
            break;
        }
    } while (currentResidualError > convergenceCriteria);

    // Copy solved vertices from GPU into the solution
    kernel.pullVertices();

    std::cout << "\nLOD " << lod << " convereged after " << totalSteps << " steps with residual " << currentResidualError << std::endl;

    return totalSteps;
}

void ProblemInstance::computeMaterialConfigurationEquationsForLOD(int lod) {
    solutionLODs[lod]->computeMaterialConfigurationEquations();
}

void ProblemInstance::debug_outputTotalForceForEachLOD() {
    int numLODs = (int)problemLODs.size() - 1;
    for (int i = numLODs; i >= 0; i--) {
        REAL totalForce = 0;

        std::unordered_map<unsigned int, NeumannBoundary>* boundaries = problemLODs.at(i)->getNeumannBoundaryMap();
        for (auto itt = boundaries->begin(); itt != boundaries->end(); itt++) {
            totalForce += asREAL(itt->second.force.getLength());
        }

        std::cout << "Total Neumann force for LOD " << i << " is " << totalForce << std::endl;
    }
}
