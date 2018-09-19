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

#define ACTIVE_DEVICE 0

void solveGPU(ProblemInstance& problemInstance, int lod) {

    WaveSampler sampler(problemInstance.getSolutionLOD(lod));
    sampler.setWaveOrientation(libmmv::Vec3ui(0, 0, 0), libmmv::Vec3ui(0, 0, 1));

    VTKSolutionVisualizer visualizer(problemInstance.getSolutionLOD(lod));
	visualizer.filterOutNullVoxels(false);

    VTKSamplingVisualizer samplingVis(problemInstance.getSolutionLOD(lod));

    VTKImportanceVisualizer impVis(problemInstance.getProblemLOD(lod), problemInstance.getResidualVolumeLOD(lod));

    SolveDisplacementKernel kernel(problemInstance.getSolutionLOD(lod), &sampler, problemInstance.getResidualVolumeLOD(lod));

    std::cout << "Updating with GPU...\n";
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 1001;

    for (int i = 1; i <= 5; i++) {
        
        //if (i % 100 == 0) {
        //    REAL totalResidual = problemInstance.getResidualVolumeLOD(lod)->getTotalResidual();
        //    std::cout << "Total residual : " << totalResidual << std::endl;

        //    kernel.pullVertices();

        //    //if (lod > 0) {
        //        std::stringstream fp;
        //        fp << "d:\\tmp\\lod" << lod << "_gpu_" << i << ".vtk";
        //        visualizer.writeToFile(fp.str());
        //    //}

        //    if (totalResidual < 1e-15) {
        //        break;
        //    }
        //}

        std::cout << ".";

        kernel.launch();

		if (false) {
            kernel.pullVertices();
            std::cout << std::endl;
			std::stringstream fp;
			fp << "d:\\tmp\\step_gpu_" << i << ".vtk";
			visualizer.writeToFile(fp.str());

            fp = std::stringstream();
            fp << "d:\\tmp\\step_sampling_" << i << ".vtk";
            samplingVis.writeToFile(fp.str(), kernel.debugGetImportanceSamplesManaged(), 64, BLOCK_SIZE);

            //fp = std::stringstream();
            //fp << "d:\\tmp\\residuals_" << i << "_";
            //impVis.writeAllLevels(fp.str());
		}
    }

    kernel.pullVertices();
    auto finish = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << std::endl << "GPU execution time: " << cpuTime.count() << " ms for " << iterations <<" iterations\n\n";

    //int i = 1;
    //std::cout << std::endl;
    //std::stringstream fp;
    //impVis.writeAllLevels("c:\\tmp\\end_residual");

    //fp = std::stringstream();
   // fp << "c:\\tmp\\lod_" << lod << "_sampling_" << i << ".vtk";
   // samplingVis.writeToFile(fp.str(), kernel.debugGetImportanceSamplesManaged(), 512, 8);

    std::stringstream fp = std::stringstream();
	fp << "d:\\tmp\\lod_" << lod << "_gpu_end.vtk";
	visualizer.writeToFile(fp.str());

    fp = std::stringstream();
    fp << "d:\\tmp\\step_sampling_" << lod << ".vtk";
    samplingVis.writeToFile(fp.str(), kernel.debugGetImportanceSamplesManaged(), 64, BLOCK_SIZE);
    
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

    bool isStumpMRC = false;
    unsigned char matFilter = 255;
    std::string filename("voxel_64.mrc");
    ProblemInstance problemInstance;
    problemInstance.initFromMaterialProbeMRC(filename);

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    BoundaryProjector bProjector(problemInstance.getProblemLOD(0));
    bProjector.setMaxProjectionDepth(5, 5);
    bProjector.projectDirichletBoundaryAlongNegZ(&fixed);

    REAL totalNeumannStressNewtons = asREAL(1);
    bProjector.projectNeumannStressAlongPosZ(totalNeumannStressNewtons, matFilter);

    // Re-initialize the residual volume for LOD 0 because we've added the boundary conditions now
    //problemInstance.getResidualVolumeLOD(0)->initializePyramidFromProblem();

    problemInstance.createAdditionalLODs(2);
    
    auto start = std::chrono::high_resolution_clock::now();
    //solveCPU(problem);
    auto finish = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
    std::cout << std::endl << "Total CPU execution time: " << cpuTime.count() << " seconds\n\n";

    solveGPU(problemInstance, 2);
    problemInstance.projectCoarseSolutionToFinerSolution(2, 1);
    solveGPU(problemInstance, 1);
    problemInstance.projectCoarseSolutionToFinerSolution(1, 0);
    solveGPU(problemInstance, 0);

    std::cout << std::endl << "GPU execution time: " << cpuTime.count() << " seconds\n\n";

}
