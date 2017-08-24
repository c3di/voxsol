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

    ettention::Vec3ui size(3, 3, 3);
    ettention::Vec3d voxelSize(1, 1, 1);

    MaterialFactory mFactory;
    MaterialDictionary mDictionary;

    DirichletBoundary fixed(DirichletBoundary::FIXED_ALL);
    NeumannBoundary stress(ettention::Vec3<REAL>(9999, 0, 0));

    Material steel = mFactory.createMaterialWithProperties(asREAL(210e9), asREAL(0.3));
    mDictionary.addMaterial(steel);

    DiscreteProblem problem(size, voxelSize, &mDictionary);

    for (int i = 0; i < 27; i++) {
        problem.setMaterial(i, steel.id);
        if (i > 18) {
            problem.setDirichletBoundaryAtVertex(i, fixed);
        }
    }

    problem.setNeumannBoundaryAtVertex(ettention::Vec3ui(3, 3, 3), stress);

    Solution solution(problem);
    solution.computeMaterialConfigurationEquations();

    SolveDisplacementKernel kernel(&solution);
    kernel.launch();

    std::vector<Vertex>* vertices = solution.getVertices();
    std::cout << "\nEquation Ids: " << std::endl;
    for (int i = 0; i < vertices->size(); i++) {
        std::cout << vertices->at(i).materialConfigId << " ";
    }
    std::cout << std::endl;
    std::cout << "\nDisplacements: " << std::endl;
    for (int i = 0; i < vertices->size(); i++) {
        Vertex v = vertices->at(i);
        std::cout << v.x << ", " << v.y << ", " << v.z << std::endl;
    }

}
