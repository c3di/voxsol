#pragma once
#include <stdafx.h>
#include <cuda_runtime.h>

class CudaKernel {
public:
    CudaKernel() {};
    ~CudaKernel() {};
    
    virtual void execute() = 0;

protected:

    // Check all execution requirements (eg. all inputs supplied, output location supplied)
    virtual bool canExecute() = 0;
    
    // Frees all device memory that was alloc'd by this kernel
    virtual void freeCudaResources() = 0;

};
