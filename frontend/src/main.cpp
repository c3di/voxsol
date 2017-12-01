#include <stdafx.h>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <algorithm>
#include "gpu/CudaDebugHelper.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "libmmv/math/Vec3.h"
#include "gpu/kernels/SolveDisplacementKernel.h"
#include "material/MaterialFactory.h"
#include "material/MaterialDictionary.h"
#include "problem/DirichletBoundary.h"
#include "problem/NeumannBoundary.h"
#include "io/VTKSolutionVisualizer.h"

#define ACTIVE_DEVICE 1

void solveCPU(DiscreteProblem& problem) {
    Solution solution(problem);
    solution.computeMaterialConfigurationEquations();

    VTKSolutionVisualizer visualizer(&solution);
    visualizer.writeToFile("c:\\tmp\\step_0.vtk");
    SolveDisplacementKernel kernel(&solution);

    std::cout << "Solving 20 * 183,600 updates with CPU...";

    for (int i = 0; i < 20; i++) {
        std::cout << " " << i;
        kernel.solveCPU();
        std::stringstream fp;
        fp << "c:\\tmp\\step_cpu_" << i << ".vtk";
        visualizer.writeToFile(fp.str());
    }

    std::cout << " DONE" << std::endl << std::endl;

    kernel.debugOutputEquationsCPU();
    kernel.debugOutputEquationsGPU();
}

void solveGPU(DiscreteProblem& problem) {
    Solution solution(problem);
    solution.computeMaterialConfigurationEquations();

    VTKSolutionVisualizer visualizer(&solution);
    visualizer.writeToFile("c:\\tmp\\step_0.vtk");
    SolveDisplacementKernel kernel(&solution);

    std::cout << "Solving 40 * 4,480,000 updates with GPU...";

    for (int i = 0; i < 400; i++) {
        std::cout << " " << i;
        kernel.launch();
        std::stringstream fp;
        fp << "c:\\tmp\\step_gpu_" << i << ".vtk";
        visualizer.writeToFile(fp.str());
    }

    std::cout << " DONE" << std::endl << std::endl;

    kernel.debugOutputEquationsCPU();
    kernel.debugOutputEquationsGPU();
}

int main(int argc, char* argv[]) {
    _putenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC=1");
    std::cout << "Stochastic Mechanic Solver -- BY OUR GPUS COMBINED!\n\n";

    CudaDebugHelper::PrintDeviceInfo(ACTIVE_DEVICE);
    CudaDebugHelper::PrintDevicePeerAccess(0, 1);

    if (CudaDebugHelper::DevicePeerAccessSupported(0, 1) && CudaDebugHelper::DevicePeerAccessSupported(1, 0)) {
        //Needed for managed memory to work properly, allow GPUs to access each other's memory
        cudaDeviceEnablePeerAccess(0, 0);
        cudaDeviceEnablePeerAccess(1, 0);
    }
    else {
        std::cout << "WARNING: Peer access is not supported by the available GPUs. Forcing managed memory to be allocated on-device. \n\n";
    }
    
    cudaSetDevice(ACTIVE_DEVICE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not initialize CUDA context: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    else {
        std::cout << "Cuda device " << ACTIVE_DEVICE << " initialized!\n\n";
    }

    ettention::Vec3ui size(100, 10, 10);
    ettention::Vec3d voxelSize(0.1, 0.1, 0.1);

    MaterialFactory mFactory;
    MaterialDictionary mDictionary;

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    std::vector<NeumannBoundary> boundaries;

    Material steel = mFactory.createMaterialWithProperties(asREAL(210e9), asREAL(0.3));
    mDictionary.addMaterial(steel);

    DiscreteProblem problem(size, voxelSize, &mDictionary);

    for (unsigned int i = 0; i < size.z*size.y*size.x; i++) {
        problem.setMaterial(i, steel.id);
    }

    for (unsigned int z = 0; z <= size.z; z++) {
        for (unsigned int y = 0; y <= size.y; y++) {
            for (unsigned int x = 0; x <= size.x; x++) {
                if (z == size.z) {
                    REAL xFactor = 1;
                    REAL yFactor = 1;
                    if (x == 0 || x == size.x) {
                        xFactor = static_cast<REAL>(0.5);
                    }
                    if (y == 0 || y == size.y) {
                        yFactor = static_cast<REAL>(0.5);
                    }
                    NeumannBoundary stress(ettention::Vec3<REAL>(0, 0, static_cast<REAL>(-1e7 * xFactor * yFactor)));
                    boundaries.push_back(stress);
                    problem.setNeumannBoundaryAtVertex(ettention::Vec3ui(x, y, z), stress);
                }
                if (x == 0) {
                    problem.setDirichletBoundaryAtVertex(ettention::Vec3ui(0, y, z), fixed);
                }
            }
        }
    }

    
    auto start = std::chrono::high_resolution_clock::now();
   // solveCPU(problem);
    auto finish = std::chrono::high_resolution_clock::now();
    auto cpuTime = finish - start;
    std::cout << std::endl << "CPU execution time: " << cpuTime.count() << " seconds\n\n";
    
    start = std::chrono::high_resolution_clock::now();
    solveGPU(problem);
    finish = std::chrono::high_resolution_clock::now();
    cpuTime = finish - start;
    std::cout << std::endl << "GPU execution time: " << cpuTime.count() << " seconds\n\n";

}
