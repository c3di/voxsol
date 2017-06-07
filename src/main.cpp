#include <cstdio>
#include <iostream>
#include "gpu/CudaDebugHelper.h"

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

}
