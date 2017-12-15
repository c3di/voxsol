#pragma once

#include "libmmv/io/rangetransformation/RangeTransformation.h"

namespace libmmv
{
    class Volume;

    class LinearRangeTransformation : public RangeTransformation
    {
    public:
        LinearRangeTransformation(Volume* volume, float outputRangeMinimum, float outputRangeMaximum);

        virtual float transformRange(float value) override;
        virtual void transformRange(float* data, size_t size) override;

    protected:
        float outMin;
        float outMax;
    };
};
