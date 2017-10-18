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

void solveCPU(DiscreteProblem& problem) {
    Solution solution(problem);
    solution.computeMaterialConfigurationEquations();

    VTKVisualizer visualizer(&solution);
    visualizer.writeToFile("c:\\tmp\\step_0.vtk");
    SolveDisplacementKernel kernel(&solution);

    for (int i = 0; i < 1; i++) {
        kernel.solveCPU();
        std::stringstream fp;
        fp << "c:\\tmp\\step_cpu.vtk";
        visualizer.writeToFile(fp.str());
    }

    kernel.debugOutputEquationsCPU();
    kernel.debugOutputEquationsGPU();
}

void solveGPU(DiscreteProblem& problem) {
    Solution solution(problem);
    solution.computeMaterialConfigurationEquations();

    VTKVisualizer visualizer(&solution);
    visualizer.writeToFile("c:\\tmp\\step_0.vtk");
    SolveDisplacementKernel kernel(&solution);

    for (int i = 0; i < 1; i++) {
        kernel.launch();
        std::stringstream fp;
        fp << "c:\\tmp\\step_gpu.vtk";
        visualizer.writeToFile(fp.str());
    }
    kernel.debugOutputEquationsCPU();
    kernel.debugOutputEquationsGPU();
}

int main(int argc, char* argv[]) {

    std::cout << "Stochastic Mechanic Solver -- BY OUR GPUS COMBINED!\n\n";
    cudaSetDevice(0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Could not initialize CUDA context: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    else {
        std::cout << "Cuda device initialized!\n\n";
    }

    CudaDebugHelper::PrintDeviceInfo(0);

    ettention::Vec3ui size(2, 2, 2);
    ettention::Vec3d voxelSize(1, 1, 1);

    MaterialFactory mFactory;
    MaterialDictionary mDictionary;

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    std::vector<NeumannBoundary> boundaries;

    Material steel = mFactory.createMaterialWithProperties(asREAL(210e9), asREAL(0.3));
    mDictionary.addMaterial(steel);

    DiscreteProblem problem(size, voxelSize, &mDictionary);

    for (int i = 0; i < 8; i++) {
        problem.setMaterial(i, steel.id);
    }

    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                if (x == 2) {
                    REAL zFactor = 1;
                    REAL yFactor = 1;
                    if (z == 0 || z == 2) {
                        zFactor = static_cast<REAL>(0.5);
                    }
                    if (y == 0 || y == 2) {
                        yFactor = static_cast<REAL>(0.5);
                    }
                    NeumannBoundary stress(ettention::Vec3<REAL>(static_cast<REAL>(1e11 * zFactor * yFactor), 0, 0));
                    boundaries.push_back(stress);
                    problem.setNeumannBoundaryAtVertex(ettention::Vec3ui(2, y, z), stress);
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
