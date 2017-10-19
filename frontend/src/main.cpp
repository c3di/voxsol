#include <stdafx.h>
#include <cstdio>
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
#include "io/VTKVisualizer.h"

#define ACTIVE_DEVICE 1

void solveCPU(DiscreteProblem& problem) {
    Solution solution(problem);
    solution.computeMaterialConfigurationEquations();

    VTKVisualizer visualizer(&solution);
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

    VTKVisualizer visualizer(&solution);
    visualizer.writeToFile("c:\\tmp\\step_0.vtk");
    SolveDisplacementKernel kernel(&solution);

    std::cout << "Solving 40 * 1.92 million updates with GPU...";

    for (int i = 0; i < 40; i++) {
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

    std::cout << "Stochastic Mechanic Solver -- BY OUR GPUS COMBINED!\n\n";

    CudaDebugHelper::PrintDeviceInfo(0);
    CudaDebugHelper::PrintDeviceInfo(1);

    cudaSetDevice(ACTIVE_DEVICE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not initialize CUDA context: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    else {
        std::cout << "Cuda device "<< ACTIVE_DEVICE << " initialized!\n\n";
    }

    ettention::Vec3ui size(50, 5, 5);
    ettention::Vec3d voxelSize(0.2, 0.2, 0.2);

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

    solveCPU(problem);
    solveGPU(problem);

}
