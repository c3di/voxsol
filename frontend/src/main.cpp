#include <stdafx.h>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <algorithm>
#include "gpu/CudaDebugHelper.h"
#include "gpu/CudaCommonFunctions.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "libmmv/math/Vec3.h"
#include "gpu/kernels/SolveDisplacementKernel.h"
#include "material/MaterialFactory.h"
#include "material/MaterialDictionary.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "io/VTKSolutionVisualizer.h"
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

#define ACTIVE_DEVICE 0
#define ENABLE_VTK_OUTPUT 1

int totalIterations = 0;
int totalMilliseconds = 0;

void solveGPU(ProblemInstance& problemInstance, int lod) {

    WaveSampler sampler(problemInstance.getSolutionLOD(lod));
    sampler.setWaveOrientation(libmmv::Vec3ui(0, 0, 0), libmmv::Vec3ui(0, 0, 1));

    VTKSolutionVisualizer visualizer(problemInstance.getSolutionLOD(lod));
	visualizer.filterOutNullVoxels(false);

    VTKSamplingVisualizer samplingVis(problemInstance.getSolutionLOD(lod));

    VTKImportanceVisualizer impVis(problemInstance.getProblemLOD(lod), problemInstance.getResidualVolumeLOD(lod));

    SolveDisplacementKernel kernel(problemInstance.getSolutionLOD(lod), &sampler, problemInstance.getResidualVolumeLOD(lod));
    
    std::cout << "Solving LOD " << lod << " with GPU...\n";
    long targetMilliseconds = 10000 / (lod+1);

    auto now = std::chrono::high_resolution_clock::now();
    __int64 elapsed = 0;
    int numIterationsDone = 0;
    int numIterationsTarget = 10 / (lod+1);

    //for (int i = 1; i <= numIterationsTarget; i++) {
    while (elapsed < targetMilliseconds) {
        auto start = std::chrono::high_resolution_clock::now();
        totalIterations++;
        std::cout << ".";

        kernel.launch();

        now = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        numIterationsDone++;
    }

    kernel.pullVertices();

    std::cout << "\nFinished simulating for " << targetMilliseconds << "ms. Did " << numIterationsDone << " iterations\n\n";
    totalMilliseconds += targetMilliseconds;

#ifdef ENABLE_VTK_OUTPUT
    std::stringstream fp = std::stringstream();
	fp << "d:\\tmp\\lod_" << lod << "_gpu_end.vtk";
	visualizer.writeToFile(fp.str());

    fp = std::stringstream();
    fp << "d:\\tmp\\step_sampling_" << lod << ".vtk";
    samplingVis.writeToFile(fp.str(), kernel.debugGetImportanceSamplesManaged(), 64, BLOCK_SIZE);
#endif 
    std::cout << " DONE. Total residual: " << problemInstance.getResidualVolumeLOD(lod)->getTotalResidual() << std::endl << std::endl;
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

    std::string xmlInputFile("voxel_64.xml");

    XMLProblemDeserializer xmlDeserializer(xmlInputFile);
    ProblemInstance problemInstance = xmlDeserializer.getProblemInstance();

    int numLODs = problemInstance.getNumberOfLODs();
    for (int lod = numLODs-1; lod >= 0; lod--) {
        solveGPU(problemInstance, lod);
        if (lod > 0) {
            problemInstance.projectCoarseSolutionToFinerSolution(lod, lod - 1);
        }
    }

    std::cout << std::endl << "Total simulation time: " << totalMilliseconds << " ms\n\n";

}
