#include <stdafx.h>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <Windows.h>
#include "io/CommandLine.h"
#include "gpu/CudaDebugHelper.h"
#include "io/XMLProblemDeserializer.h"
#include "io/VTKSolutionWriter.h"
#include "io/VTKImportanceVisualizer.h"
#include "io/CSVSolutionWriter.h"
#include "solution/samplers/SequentialBlockSampler.h"
#include "gpu/kernels/SolveDisplacementKernel.h"


#define ACTIVE_DEVICE 0

int totalIterations = 0;
__int64 elapsedMicroseconds = 0;
HWND myConsoleWindow = GetConsoleWindow();

REAL EPSILON = asREAL(1.0e-6);

bool OUTPUT_VTK = true;
bool OUTPUT_CSV = false;
int RESIDUAL_UPDATE_FREQUENCY = 500;
int PERIODIC_OUTPUT_FREQUENCY = -1;
bool DEBUG_OUTPUT = true;

int nextIterationTarget = 1;

std::string base_file_path = "";

void solveGPU(ProblemInstance& problemInstance, int lod) {
    SequentialBlockSampler sampler(problemInstance.getSolutionLOD(lod), 6);

    SolveDisplacementKernel kernel(problemInstance.getSolutionLOD(lod), &sampler, problemInstance.getResidualVolumeLOD(lod));

    VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(lod), problemInstance.getResidualVolumeLOD(lod));
    vtkWriter.filterOutNullVoxels();
    vtkWriter.setMechanicalValuesOutput(true);

    auto now = std::chrono::high_resolution_clock::now();
    int numIterationsDone = 0;
    int numVerticesNotConverged = (int)problemInstance.getSolutionLOD(lod)->getVertices()->size();
    int numVerticesOnResidualLevelZero = (int)problemInstance.getResidualVolumeLOD(lod)->getNumVerticesOnLevelZero();
    REAL remainingResidual = 10000000;
    kernel.setNumLaunchesBeforeResidualUpdate(RESIDUAL_UPDATE_FREQUENCY - 1);
    libmmv::Vec3<REAL> voxelSize = problemInstance.getSolutionLOD(lod)->getVoxelSize();
    REAL maxVoxelDim = std::max(std::max(voxelSize.x, voxelSize.y), voxelSize.z);

    REAL targetResid = std::max(maxVoxelDim * EPSILON, asREAL(1.0e-7));

    std::cout << "Solving LOD " << lod << " with GPU, target residual " << EPSILON << "\n";

    while (remainingResidual > EPSILON) {
        auto start = std::chrono::high_resolution_clock::now();
        totalIterations++;

        kernel.launch();

        now = std::chrono::high_resolution_clock::now();
        elapsedMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        numIterationsDone++;

        if (totalIterations % RESIDUAL_UPDATE_FREQUENCY == 0) {
            remainingResidual = problemInstance.getResidualVolumeLOD(lod)->getResidualDeltaToLastUpdate(&numVerticesNotConverged);
            std::cout << "\rResidual: " << remainingResidual << " [" << numVerticesNotConverged << "/" << numVerticesOnResidualLevelZero << "] iteration " << totalIterations << "                 ";
            
            if (isnan(remainingResidual)) {
                std::cout << "\n\n\t[ERROR] NAN residual found, outputting snapshot and aborting simulation\n\n";

                VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(lod));
                vtkWriter.filterOutNullVoxels();
                vtkWriter.setMechanicalValuesOutput(true);
                kernel.pullVertices();
                std::stringstream fp = std::stringstream();
                fp << base_file_path << "lod_" << lod << "_NAN_snapshot_" << totalIterations << ".vtk";
                vtkWriter.writeEntireStructureToFile(fp.str());

                exit(EXIT_FAILURE);
            }
        }

        // Output periodic snapshots of the solution
        if (PERIODIC_OUTPUT_FREQUENCY > 0 && (numIterationsDone == 1 || totalIterations == nextIterationTarget)) {
            kernel.pullVertices();

            if (!DEBUG_OUTPUT) {
                for (int i = lod; i > 0; i--) {
                    problemInstance.projectCoarseSolutionToFinerSolution(i, i - 1);
                }

                VTKSolutionWriter vtkSnapshot(problemInstance.getSolutionLOD(0), problemInstance.getResidualVolumeLOD(0));
                vtkSnapshot.filterOutNullVoxels();
                vtkSnapshot.setMechanicalValuesOutput(true);
                std::stringstream fp = std::stringstream();
                fp << base_file_path << "iteration_" << totalIterations << ".vtk";
                vtkSnapshot.writeEntireStructureToFile(fp.str());
            }
            else {
                std::stringstream fp = std::stringstream();
                fp << base_file_path << "iteration_" << totalIterations << ".vtk";
                vtkWriter.writeEntireStructureToFile(fp.str());
            }
            
            nextIterationTarget += PERIODIC_OUTPUT_FREQUENCY;
        }

        if (GetKeyState('Q') & 0x8000/*Check if high-order bit is set (1 << 15)*/ && GetForegroundWindow() == myConsoleWindow)
        {
            std::cout << "\nStopping here..." << std::endl;
            Sleep(500);
            break;
        }

        if (GetKeyState('S') & 0x8000/*Check if high-order bit is set (1 << 15)*/ && GetForegroundWindow() == myConsoleWindow)
        {
            VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(lod), problemInstance.getResidualVolumeLOD(lod));
            vtkWriter.filterOutNullVoxels();
            vtkWriter.setMechanicalValuesOutput(true);

            std::cout << "\nOutputting snapshot of current LOD...";

            kernel.pullVertices();
            std::stringstream fp = std::stringstream();
            fp << base_file_path << "lod_" << lod << "_snapshot_" << totalIterations << ".vtk";
            vtkWriter.writeEntireStructureToFile(fp.str());

            VTKImportanceVisualizer impVis(problemInstance.getProblemLOD(lod), problemInstance.getResidualVolumeLOD(lod));
            fp = std::stringstream();
            fp << base_file_path << "lod_" << lod << "_residuals_" << totalIterations << ".vtk";
            impVis.writeToFile(fp.str(), 0);

            std::cout << "Done.\n";
        }
    }

    kernel.pullVertices();
    std::cout << "\nFinished simulating for " << elapsedMicroseconds / 1000 << " ms. Did " << numIterationsDone << " iterations\n\n";

}

int main(int argc, char* argv[]) {
    _putenv("CUDA_VISIBLE_DEVICES=0");
    std::cout << "Stochastic Mechanic Solver\n\n";

    CudaDebugHelper::PrintDeviceInfo(ACTIVE_DEVICE);

    cudaSetDevice(ACTIVE_DEVICE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not initialize CUDA context: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    else {
        std::cout << "Cuda device " << ACTIVE_DEVICE << " initialized!\n\n";
    }

    CommandLine clParser(argc, argv);
    const std::string &xmlFilename = clParser.getCmdOption("-i");

    std::string xmlInputFile("test.xml");

    if (xmlFilename.empty()) {
        std::cout << "No input filename found in command line options, using " << xmlInputFile << " instead.\n";
    }
    else {
        xmlInputFile = xmlFilename;
    }

    const std::string outputFolder = clParser.getCmdOption("-o");
    if (!outputFolder.empty()) {
        base_file_path = outputFolder;
    }

    const std::string periodicSnapshots = clParser.getCmdOption("-s");
    if (!periodicSnapshots.empty()) {
        try {
            int rate = std::stoi(periodicSnapshots);
            PERIODIC_OUTPUT_FREQUENCY = rate;
        }
        catch (std::exception const e) {
            std::cout << "Invalid command line argument: -s " << periodicSnapshots << std::endl;
        }
    }
    const std::string outputOptions = clParser.getCmdOption("--outputs");
    if (!outputOptions.empty()) {
        OUTPUT_VTK = outputOptions.find("vtk") != -1 || outputOptions.find("VTK") != -1;
        OUTPUT_CSV = outputOptions.find("csv") != -1 || outputOptions.find("CSV") != -1;
    }

    XMLProblemDeserializer xmlDeserializer(xmlInputFile);
    std::unique_ptr<ProblemInstance> problemInstance = xmlDeserializer.getProblemInstance();
    problemInstance->computeMaterialConfigurationEquations();

    EPSILON = xmlDeserializer.getTargetResidual();

    std::cout << "\nUsing target residual " << EPSILON << "\n";

    int numLODs = problemInstance->getNumberOfLODs();
    for (int lod = numLODs-1; lod >= 0; lod--) {
        solveGPU(*problemInstance, lod);
        if (lod > 0) {
            problemInstance->projectCoarseSolutionToFinerSolution(lod, lod - 1);
        }
    }

    std::cout << std::endl << "Total simulation time: " << elapsedMicroseconds / 1000 << " ms\n\n";

    try {
        // Remove any directories and the .xml ending
        std::string filename = xmlInputFile.substr(xmlInputFile.find_last_of("/\\") + 1);
        std::string::size_type const p(filename.find_last_of('.'));
        filename = filename.substr(0, p);

        std::stringstream fp = std::stringstream();

        if (OUTPUT_CSV) {
            CSVSolutionWriter csvWriter(problemInstance->getSolutionLOD(0));
            fp << base_file_path << filename << ".csv";
            csvWriter.writeSolutionToFile(fp.str());
        }

        if (OUTPUT_VTK) {
            VTKSolutionWriter vtkWriter(problemInstance->getSolutionLOD(0));
            vtkWriter.filterOutNullVoxels();
            vtkWriter.setMechanicalValuesOutput(true);
            
            fp = std::stringstream();
            fp << base_file_path << filename << ".vtk";
            vtkWriter.writeEntireStructureToFile(fp.str());
        }
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
