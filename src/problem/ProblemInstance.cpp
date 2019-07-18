#include "stdafx.h"
#include <iomanip>
#include "ProblemInstance.h"
#include "io/MRCVoxelImporter.h"
#include "io/MRCImporter.h"
#include "tools/LODGenerator.h"
#include "gpu/kernels/SolveDisplacementKernel.h"
#include "gpu/sampling/ImportanceBlockSampler.h"
#include "solution/samplers/SequentialBlockSampler.h"
#include "io/VTKSamplingVisualizer.h"
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

void ProblemInstance::initFromMRCStack(std::string& path, bool isStumpMRC) {
    std::cout << "\nImporting MRC stack using configuration: " << (isStumpMRC ? " STUMP " : " IMPLANT ") << std::endl;

    MRCVoxelImporter importer(path);
    materialDictionary = importer.extractMaterialDictionary();

    if (isStumpMRC) {
        importer.fatMaterialColorValue = 1;
        importer.muscleMaterialColorValue = 2;
        importer.boneMaterialColorValue = 3;
        importer.skinMaterialColorValue = 5;
        importer.linerMaterialColorValue = 7;
        importer.socketMaterialColorValue = 11;
    }

    libmmv::Vec3<REAL> voxelSize(0, 0, 0);
    if (isStumpMRC) {
        voxelSize.x = asREAL(0.0024);
        voxelSize.y = asREAL(0.0024); 
        voxelSize.z = asREAL(0.0030);
    }
    else {
        voxelSize.x = asREAL(0.0024);
        voxelSize.y = asREAL(0.0024);
        voxelSize.z = asREAL(0.014);
    }

    DiscreteProblem* problem = new DiscreteProblem(importer.getDimensionsInVoxels(), voxelSize, &materialDictionary);
    importer.populateDiscreteProblem(problem);
    problemLODs.push_back(problem);

    Solution* solution = new Solution(problemLODs[0]);
    solutionLODs.push_back(solution);
    
    ResidualVolume* residual = new ResidualVolume(problemLODs[0]);
    residualLODs.push_back(residual);
}

void ProblemInstance::initFromMaterialProbeMRC(std::string & path) {
    std::cout << "\nImporting material probe MRC stack..." << std::endl;

    MRCImporter importer(path);

    Material aluminum = materialFactory.createMaterialWithProperties(asREAL(70000000000), asREAL(0.35));
    Material silicon = materialFactory.createMaterialWithProperties(asREAL(150000000000), asREAL(0.27));
    materialDictionary.addMaterial(aluminum);
    materialDictionary.addMaterial(silicon);

    importer.addMaterialMapping(&aluminum, 0);
    importer.addMaterialMapping(&silicon, 255);

    libmmv::Vec3ui voxelDims = importer.getDimensionsInVoxels();

    libmmv::Vec3<REAL> voxelSize(asREAL(1.0 / voxelDims.x), asREAL(1.0 / voxelDims.y), asREAL(1.0 / voxelDims.z));

    DiscreteProblem* problem = new DiscreteProblem(voxelDims, voxelSize, &materialDictionary);
    importer.populateDiscreteProblem(problem);
    problemLODs.push_back(problem);

    Solution* solution = new Solution(problemLODs[0]);
    solutionLODs.push_back(solution);

    ResidualVolume* residual = new ResidualVolume(problemLODs[0]);
    residualLODs.push_back(residual);
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
    computeMaterialConfigurationEquationsForLOD(0);
    getResidualVolumeLOD(0)->initializePyramidFromProblem();

    LODGenerator lodGen;
    libmmv::Vec3ui size = problemLODs[0]->getSize();
    libmmv::Vec3<REAL> voxelSize = problemLODs[0]->getVoxelSize();
    for (int fine = 0; fine < numberOfAdditionalLODs; fine++) {
        int coarse = fine + 1;

        if (size.x > 1) {
            size.x /= 2;
            voxelSize.x *= 2;
        }
        if (size.y > 1) {
            size.y /= 2;
            voxelSize.y *= 2;
        }
        if (size.z > 1) {
            size.z /= 2;
            voxelSize.z *= 2;
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

        computeMaterialConfigurationEquationsForLOD(coarse);
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

    VTKSamplingVisualizer samplingVis(getSolutionLOD(lod));
    VTKSolutionWriter vis(getSolutionLOD(lod));
    
    REAL currentResidualError = 1000000;
    int totalSteps = 0;
    do {
        kernel.launch();

        totalSteps++;

        currentResidualError = residualVolume->getAverageResidual(convergenceCriteria);

        if (totalSteps % 500 == 0 || currentResidualError <= convergenceCriteria) {
            std::cout << "\rCurrent residual: " << currentResidualError << "                                         ";
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

