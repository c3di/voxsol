#include <stdafx.h>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "gpu/CudaDebugHelper.h"
#include "solution/MatrixPrecomputer.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "libmmv/math/Vec3.h"
#include "gpu/kernels/CK_SolveDisplacement.h"

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

    DiscreteProblem problem(size, voxelSize);
    Material steel(210e9, 0.3);

    for (int i = 0; i < 27; i++) {
        problem.setMaterial(i, steel);
    }

    Solution solution(problem);
    solution.precomputeMatrices();

    CK_SolveDisplacement kernel(&solution);
    kernel.launchKernel();


}
