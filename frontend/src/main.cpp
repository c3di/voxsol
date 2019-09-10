#include <stdafx.h>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <Windows.h>
#include "gpu/CudaDebugHelper.h"
#include "gpu/GPUParameters.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "libmmv/math/Vec3.h"
#include "gpu/kernels/SolveDisplacementKernel.h"
#include "material/MaterialFactory.h"
#include "material/MaterialDictionary.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "io/VTKSolutionWriter.h"
#include "io/VTKSolutionStructuredWriter.h"
#include "io/VTKImportanceVisualizer.h"
#include "io/VTKSamplingVisualizer.h"
#include "io/MRCVoxelImporter.h"
#include "problem/boundaryconditions/BoundaryProjector.h"
#include "gpu/sampling/ResidualVolume.h"
#include "gpu/sampling/ImportanceBlockSampler.h"
#include "gpu/sampling/WaveSampler.h"
#include "problem/ProblemInstance.h"
#include "material/MaterialConfiguration.h"
#include "io/XMLProblemDeserializer.h"
#include "solution/samplers/SequentialBlockSampler.h"

#define ACTIVE_DEVICE 0
//#define ENABLE_VTK_OUTPUT 1

int totalIterations = 0;
int totalMilliseconds = 0;

HWND myConsoleWindow = GetConsoleWindow();

void solveGPU(ProblemInstance& problemInstance, int lod) {
    //ImportanceBlockSampler sampler(problemInstance.getResidualVolumeLOD(lod));
    libmmv::Vec3i waveOrigin(0, 0, problemInstance.getSolutionLOD(lod)->getSize().z);
    libmmv::Vec3i waveDirection(0, 0, -1);

    //WaveSampler sampler(problemInstance.getSolutionLOD(lod), waveOrigin, waveDirection);
    //ImportanceBlockSampler sampler(problemInstance.getResidualVolumeLOD(lod));
    SequentialBlockSampler sampler(problemInstance.getSolutionLOD(lod), 6);

    VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(lod));
    vtkWriter.filterOutNullVoxels();
    vtkWriter.setMechanicalValuesOutput(false);

    VTKSamplingVisualizer samplingVis(problemInstance.getSolutionLOD(lod));

    VTKImportanceVisualizer impVis(problemInstance.getProblemLOD(lod), problemInstance.getResidualVolumeLOD(lod));

    SolveDisplacementKernel kernel(problemInstance.getSolutionLOD(lod), &sampler, problemInstance.getResidualVolumeLOD(lod));
    
    std::cout << "Solving LOD " << lod << " with GPU...\n";
    long targetMilliseconds = 20000;

    auto now = std::chrono::high_resolution_clock::now();
    __int64 elapsed = 0;
    int numIterationsDone = 0;
    int numIterationsTarget = 200;
    REAL maxResidual = 10000000;
    REAL aveResidual = 10000000;
    REAL epsilon = asREAL(1.0e-8);
    kernel.setNumLaunchesBeforeResidualUpdate(1999);
    int numVerticesNotConverged= 0;
    int numVerticesOnResidualLevelZero = (int)problemInstance.getResidualVolumeLOD(lod)->getNumVerticesOnLevelZero();

    //std::stringstream fp = std::stringstream();

    //fp << "d:\\tmp\\gpu_start_lod" << lod << ".vtk";
    //visualizer.filterOutNullVoxels(true);
    //visualizer.writeToFile(fp.str());

    //for (int i = 1; i <= numIterationsTarget; i++) {
    //while (elapsed < targetMilliseconds) {
    while (aveResidual > epsilon) {
        auto start = std::chrono::high_resolution_clock::now();
        totalIterations++;
        
        kernel.launch();

		now = std::chrono::high_resolution_clock::now();
		elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
		numIterationsDone++;

        if (false) {
            kernel.pullVertices();
            std::stringstream fp = std::stringstream();

            fp = std::stringstream();
            fp << "d:\\tmp\\gpu_" << totalIterations << ".vtk";
            vtkWriter.writeEntireStructureToFile(fp.str());

            fp = std::stringstream();
            fp << "d:\\tmp\\step_residuals_" << totalIterations << ".vtk";
            impVis.writeToFile(fp.str(), 0);

            fp = std::stringstream();
            fp << "d:\\tmp\\step_sampling_" << totalIterations << ".vtk";
            samplingVis.writeToFile(fp.str(), kernel.debugGetImportanceSamplesManaged(), kernel.numBlockOriginsPerIteration, BLOCK_SIZE);
        }

        if (totalIterations % 2000 == 0) {
            aveResidual = problemInstance.getResidualVolumeLOD(lod)->getAverageResidualWithThreshold( epsilon, &numVerticesNotConverged);

            if (numVerticesNotConverged < numVerticesOnResidualLevelZero * 0.0001) {
                aveResidual = epsilon;
            }  
        }

        if (totalIterations % 100 == 0) {
            std::cout << "\rResidual: " << maxResidual << " [" << numVerticesNotConverged << "/" << numVerticesInSolution << "] iteration " << totalIterations << "                 ";
        }

        if (GetKeyState('Q') & 0x8000/*Check if high-order bit is set (1 << 15)*/ && GetForegroundWindow() == myConsoleWindow)
        {
            std::cout << "\nStopping here..." << std::endl;
            break;
        }
    }

    totalMilliseconds += (int)elapsed;
    kernel.pullVertices();

    std::cout << "\nFinished simulating for " << elapsed << "ms. Did " << numIterationsDone << " iterations\n\n";
    

#ifdef ENABLE_VTK_OUTPUT
    std::stringstream fp = std::stringstream();
	fp << "d:\\tmp\\lod_" << lod << "_gpu_end.vtk";
	visualizer.writeToFile(fp.str());

    //fp = std::stringstream();
    //fp << "d:\\tmp\\step_sampling_" << lod << ".vtk";
    //samplingVis.writeToFile(fp.str(), kernel.debugGetImportanceSamplesManaged(), 768, BLOCK_SIZE);

    //fp = std::stringstream();
    //fp << "d:\\tmp\\step_residuals_" << lod << ".vtk";
    //impVis.writeAllLevels(fp.str());
#endif 
}

int main(int argc, char* argv[]) {
    _putenv("CUDA_VISIBLE_DEVICES=0");
    std::cout << "Stochastic Mechanic Solver -- BY OUR GPUS COMBINED! (except currently limited to 1 GPU)\n\n";
    

    CudaDebugHelper::PrintDeviceInfo(ACTIVE_DEVICE);
    //CudaDebugHelper::PrintDevicePeerAccess(0, 1);

    /*if (CudaDebugHelper::DevicePeerAccessSupported(0, 1) && CudaDebugHelper::DevicePeerAccessSupported(1, 0)) {
        //Needed for managed memory to work properly, allow GPUs to access each other's memory
        cudaDeviceEnablePeerAccess(0, 0);
        cudaDeviceEnablePeerAccess(1, 0);
    }
    else {
        std::cout << "WARNING: Peer access is not supported by the available GPUs. Forcing managed memory to be allocated on-device. \n\n";
    }*/
    
    cudaSetDevice(ACTIVE_DEVICE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not initialize CUDA context: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    else {
        std::cout << "Cuda device " << ACTIVE_DEVICE << " initialized!\n\n";
    }

    std::string xmlInputFile("demonstrator_bone.xml");

    XMLProblemDeserializer xmlDeserializer(xmlInputFile);
    std::unique_ptr<ProblemInstance> problemInstance = xmlDeserializer.getProblemInstance();

    int numLODs = problemInstance->getNumberOfLODs();
    for (int lod = numLODs-1; lod >= 0; lod--) {
        solveGPU(*problemInstance, lod);
        if (lod > 0) {
            problemInstance->projectCoarseSolutionToFinerSolution(lod, lod - 1);
        }
    }

    std::vector<Vertex>* vertices = problemInstance->getSolutionLOD(0)->getVertices();
    REAL maxDisp = 0;
    for (auto it = vertices->begin(); it != vertices->end(); it++) {
        if (it->z > maxDisp) {
            maxDisp = it->z;
        }
    }

    std::cout << std::endl << "Maximum displacement in Z: " << maxDisp << std::endl;

    VTKSolutionWriter vtkWriter(problemInstance.getSolutionLOD(0));
    vtkWriter.filterOutNullVoxels();
    vtkWriter.setMechanicalValuesOutput(true);

    std::stringstream fp = std::stringstream();
    fp << "d:\\tmp\\gpu_end.vtk";
    vtkWriter.writeEntireStructureToFile(fp.str());
   
    std::cout << std::endl << "Total simulation time: " << totalMilliseconds << " ms\n\n";

    return 0;
}
