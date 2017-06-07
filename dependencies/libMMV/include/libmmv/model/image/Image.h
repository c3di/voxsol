#pragma once
#include <vector>
#include "math/Vec2.h"
#include "math/Vec3.h"

namespace ettention
{
    class CLMemBuffer;

    class Image
    {
    public:
        typedef Vec2ui ResolutionType;

        Image(const Image* other);
        Image(const Vec2ui& resolution, const float* initialData = 0);
        ~Image();
        
        const Vec2ui& getResolution() const;
        std::size_t getPixelCount() const;
        std::size_t getByteWidth() const;

        float* getData();
        const float* getData() const;
        float* getScanLineData(unsigned int y);
        const float* getScanLineData(unsigned int y) const;

        void convertToLogarithmicScale();
        void convertToLinearScale();

        float getPixel(int x, int y) const;
        void setPixel(size_t x, size_t y, float value);

        float computeAverageValue() const;
        float findMinValue() const;
        float findMaxValue() const;

    private:
        Vec2ui resolution;
        std::vector<float> data;
    };
}