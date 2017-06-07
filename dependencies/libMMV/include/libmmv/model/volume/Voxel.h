#pragma once

namespace ettention
{
    typedef uint8_t byte_t;
    typedef uint16_t half_float_t;

    class Voxel
    {
    public:

        enum class DataType
        {
            UCHAR_8 = 0,
            HALF_FLOAT_16 = 1,
            FLOAT_32 = HALF_FLOAT_16 << 1
        };

        static size_t sizeOfVoxelType(const DataType type);
        static std::string makeKernelDefine(const DataType type);
        static DataType parseVoxelType(const std::string value);
    };

}
