#include "gtest/gtest.h"

int main(int argc, char** argv)
{
    _putenv("CUDA_VISIBLE_DEVICES=0");
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef _WINDOWS
    system("pause");
#endif
    return result;
}
