#pragma once
#include "gpu/CudaCommonFunctions.h"

class CudaKernel {
public:
    CudaKernel() {};
    ~CudaKernel() {};
    
    virtual void launchKernel() = 0;

protected:

    // Check all execution requirements (eg. all inputs supplied, output location supplied)
    virtual bool canExecute() = 0;
    
    // Frees all device memory that was alloc'd by this kernel
    virtual void freeCudaResources() = 0;

};
