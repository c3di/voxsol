#pragma once
#include "libmmv/algorithm/volumestatistics/VolumeStatistics.h"

namespace libmmv
{
    class RangeTransformation
    {
    public:
        RangeTransformation(Volume* volume);

        virtual float transformRange(float value) = 0;
        virtual void transformRange(float* data, size_t size) = 0;

        VolumeStatistics getTransformedStatistics();

    protected:
        VolumeStatistics statistics;
    };
}
