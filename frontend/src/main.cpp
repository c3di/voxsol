#include <stdafx.h>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "gpu/CudaDebugHelper.h"
#include "solution/MatrixPrecomputer.h"
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "libmmv/math/Vec3.h"

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
    ettention::Vec3ui size(3, 2, 2);
    ettention::Vec3d voxelSize(1, 1, 1);

    DiscreteProblem problem(size, voxelSize);
    Material mat(7830, 210e9, 0.3, 0.0);
    Material mat2(6130, 200e9, 0.35, 0.0);
    problem.setMaterial(0, mat); problem.setMaterial(1, mat); problem.setMaterial(2, mat); problem.setMaterial(3, mat);
    problem.setMaterial(4, mat); problem.setMaterial(5, mat); problem.setMaterial(6, mat); problem.setMaterial(7, mat);
    problem.setMaterial(8, mat); problem.setMaterial(9, mat); problem.setMaterial(10, mat); problem.setMaterial(11, mat);

    Solution solution(problem);
    solution.precomputeMatrices();
}
