#include "gtest/gtest.h"
#include <cuda_runtime.h>

class BasicCudaTests : public ::testing::Test {

public:
    BasicCudaTests() {}
    ~BasicCudaTests() {}

    void SetUp() override
    {
        
    }

    void TearDown() override
    {
        
    }

};

TEST_F(BasicCudaTests, CudaInitTest) {

    cudaSetDevice(0);
    cudaError_t err = cudaGetLastError();
    ASSERT_TRUE(err == cudaSuccess);
 
}
