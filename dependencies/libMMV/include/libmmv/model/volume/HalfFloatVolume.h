#pragma once
#include "libmmv/math/Vec3.h"
#include "libmmv/model/volume/Volume.h"
#include "libmmv/model/volume/VolumeProperties.h"
#include "libmmv/model/volume/Voxel.h"

namespace libmmv
{
    class AlgebraicParameterSet;

    class HalfFloatVolume : public Volume
    {
    public:

        HalfFloatVolume(const Vec3ui& resolution, std::optional<float> initValue);
        HalfFloatVolume(const Vec3ui& resolution, const BoundingBox3f& bbox, std::optional<float> initValue);
        HalfFloatVolume(const Vec3ui& resolution, const float* initialData);
        ~HalfFloatVolume();

        std::unique_ptr<float> convertToFloat() override;

        void setZSlice(unsigned int z, Image* image) override;
        void setVoxelToValue(const size_t index, float value) override;
        float getVoxelValue(size_t index) const override;

    protected:
        void allocateMemory() override;
        void fillMemoryInNativeFormat(void* addr, float value, size_t count) override;
        void* getRawPointerStartingFrom(size_t offset = 0) const override;

        void init(const float* initialData);

    private:
        half_float_t *data;
    };
};
