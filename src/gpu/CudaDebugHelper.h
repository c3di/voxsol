#pragma once

#include <cuda_runtime.h>
#include <cstdio>

class CudaDebugHelper {

public:
    static void PrintDeviceInfo(int deviceId) {
        cudaDeviceProp* properties = new cudaDeviceProp();
        cudaGetDeviceProperties(properties, 0);

        printf("Properties for CUDA device %u:\n", deviceId);
        printf("\tName: \t\t\t\t%s\n", properties->name);
        printf("\tPCI device ID: \t\t\t%u\n", properties->pciDeviceID);
        printf("\tMax compute capability: \t%u.%u\n", properties->major, properties->minor);

        printf("\n  Memory\n");
        printf("\tGlobal memory: \t\t\t%zi bytes\n", properties->totalGlobalMem);
        printf("\tConstant memory: \t\t%zi bytes\n", properties->totalConstMem);
        printf("\tShared memory per block: \t%zi bytes\n", properties->sharedMemPerBlock);
        printf("\tShared memory per MP: \t\t%zi bytes\n", properties->sharedMemPerMultiprocessor);
        printf("\tBus width: \t\t\t%u bits\n", properties->memoryBusWidth);

        printf("\n  Stats\n");
        printf("\tMP count: \t\t\t%u\n", properties->multiProcessorCount);
        printf("\tMax threads per MP: \t\t%u\n", properties->maxThreadsPerMultiProcessor);
        printf("\tMax threads per block: \t\t%u\n", properties->maxThreadsPerBlock);
        printf("\tWarp size: \t\t\t%u\n", properties->warpSize);
        printf("\tMax 2D tex size: \t\t%u x %u\n", properties->maxTexture2D[0], properties->maxTexture2D[1]);
        printf("\tMax 3D tex size: \t\t%u x %u x %u\n", properties->maxTexture3D[0], properties->maxTexture3D[1], properties->maxTexture3D[2]);

        printf("\n");

        delete properties;
    }

    static void PrintDevicePeerAccess(int device1, int device2) {
        printf("Peer access is supported for:\n");
        printf("\tDevice %u --> Device %u: %s\n", device1, device2, DevicePeerAccessSupported(device1, device2) ? "Yes" : "No");
        printf("\tDevice %u --> Device %u: %s\n", device2, device1, DevicePeerAccessSupported(device2, device1) ? "Yes" : "No");
        printf("\n");
    }

    static bool DevicePeerAccessSupported(int device1, int device2) {
        bool canAccess = false;
        int* isPossible = new int;
        *isPossible = 0;
        cudaDeviceCanAccessPeer(isPossible, device1, device2);
        if (*isPossible) {
            canAccess = true;
        }
        delete isPossible;
        return canAccess;
    }
};

