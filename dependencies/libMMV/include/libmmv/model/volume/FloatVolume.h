#pragma once
#include "math/Vec3.h"
#include "model/volume/Volume.h"
#include "model/volume/VolumeProperties.h"
#include "model/volume/Voxel.h"

namespace ettention
{
    class AlgebraicParameterSet;

    class FloatVolume : public Volume
    {
    public:

        FloatVolume(const Vec3ui& resolution, boost::optional<float> initValue);
        FloatVolume(const Vec3ui& resolution, const float* initialData);
        FloatVolume(const Vec3ui& resolution, const BoundingBox3f& bbox, boost::optional<float> initValue);
        FloatVolume(const Vec3ui& resolution, const Vec3f& voxelSize, boost::optional<float> initValue);
        ~FloatVolume();

        std::unique_ptr<float> convertToFloat() override;

        void setZSlice(unsigned int z, Image* image) override;
        void setVoxelToValue(const size_t index, float value) override;
        void setVoxelToValueByCoord(const Vec3ui coords, float value);
        float getVoxelValue(size_t index) const override;
        float getVoxelValue(Vec3ui coords) const;

		inline float  nativeVoxelValue(size_t index) const { return data[index]; };
		inline float& nativeVoxelValue(size_t index) { return data[index]; };

        Image* extractSlice(unsigned int index, CoordinateOrder orientation = ORDER_XYZ) const override;

        float* getFloatPointerStartingFrom(size_t offset = 0) const;

    protected:
        void allocateMemory() override;
        void fillMemoryInNativeFormat(void* addr, float value, size_t count) override;
        void* getRawPointerStartingFrom(size_t offset = 0) const override;

        void init(const float* initialData);

    private:
        float *data;
    };
};
