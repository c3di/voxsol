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
#include "io/CSVSolutionWriter.h"
#include "solution/samplers/SequentialBlockSampler.h"
#include "gpu/kernels/SolveDisplacementKernel.h"


#define ACTIVE_DEVICE 0

int totalIterations = 0;
__int64 elapsed = 0;
HWND myConsoleWindow = GetConsoleWindow();

REAL EPSILON = asREAL(1.0e-6);
int RESIDUAL_UPDATE_FREQUENCY = 500;
int PERIODIC_OUTPUT_FREQUENCY = -1;

int nextIterationTarget = 1;

std::string base_file_path = "";

void solveGPU(ProblemInstance& problemInstance, int lod) {
    SequentialBlockSampler sampler(problemInstance.getSolutionLOD(lod), 6);

    SolveDisplacementKernel kernel(problemInstance.getSolutionLOD(lod), &sampler, problemInstance.getResidualVolumeLOD(lod));

    VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(lod));
    vtkWriter.filterOutNullVoxels();
    vtkWriter.setMechanicalValuesOutput(true);

    std::cout << "Solving LOD " << lod << " with GPU...\n";
    auto now = std::chrono::high_resolution_clock::now();
    int numIterationsDone = 0;
    int numVerticesNotConverged = 0;
    int numVerticesOnResidualLevelZero = (int)problemInstance.getResidualVolumeLOD(lod)->getNumVerticesOnLevelZero();
    REAL remainingResidual = 10000000;
    kernel.setNumLaunchesBeforeResidualUpdate(RESIDUAL_UPDATE_FREQUENCY - 1);
    
    while (remainingResidual > EPSILON) {
        auto start = std::chrono::high_resolution_clock::now();
        totalIterations++;

        kernel.launch();

        now = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
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

            std::stringstream fp = std::stringstream();
            fp << base_file_path << "iteration_" << totalIterations << ".vtk";
            vtkWriter.writeEntireStructureToFile(fp.str());

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
            VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(lod));
            vtkWriter.filterOutNullVoxels();
            vtkWriter.setMechanicalValuesOutput(true);

            std::cout << "\nOutputting snapshot of current LOD...";

            kernel.pullVertices();
            std::stringstream fp = std::stringstream();
            fp << base_file_path << "lod_" << lod << "_snapshot_" << totalIterations << ".vtk";
            vtkWriter.writeEntireStructureToFile(fp.str());

            std::cout << "Done.\n";
        }
    }

    kernel.pullVertices();
    std::cout << "\nFinished simulating for " << elapsed << "ms. Did " << numIterationsDone << " iterations\n\n";

}

int main(int argc, char* argv[]) {
    _putenv("CUDA_VISIBLE_DEVICES=0");
    std::cout << "Stochastic Mechanic Solver -- BY OUR GPUS COMBINED! (except currently limited to 1 GPU)\n\n";

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

    std::string xmlInputFile("tibia_image_update.xml");

    if (xmlFilename.empty()) {
        std::cout << "No filename found in command line options, using " << xmlInputFile << " instead.\n";
    }
    else {
        xmlInputFile = xmlFilename;
    }

    const std::string basePathOption = clParser.getCmdOption("-basePath");
    if (!basePathOption.empty()) {
        base_file_path = basePathOption;
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

    std::cout << std::endl << "Total simulation time: " << elapsed << " ms\n\n";

    VTKSolutionWriter vtkWriter(problemInstance->getSolutionLOD(0));
    vtkWriter.filterOutNullVoxels();
    vtkWriter.setMechanicalValuesOutput(true);

    CSVSolutionWriter csvWriter(problemInstance->getSolutionLOD(0));

    try {
        // Remove any directories and the .xml ending
        std::string filename = xmlInputFile.substr(xmlInputFile.find_last_of("/\\") + 1);
        std::string::size_type const p(filename.find_last_of('.'));
        filename = filename.substr(0, p);

        std::stringstream fp = std::stringstream();
        fp << base_file_path << "CSV_" << filename << ".csv";
        csvWriter.writeSolutionToFile(fp.str());

        fp = std::stringstream();
        fp << base_file_path << "RESULT_" << filename << ".vtk";
        vtkWriter.writeEntireStructureToFile(fp.str());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
