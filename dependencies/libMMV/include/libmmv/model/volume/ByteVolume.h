#pragma once
#include "libmmv/math/Vec3.h"
#include "libmmv/model/volume/Voxel.h"
#include "libmmv/model/volume/Volume.h"
#include "libmmv/model/volume/VolumeProperties.h"

namespace libmmv
{
    class AlgebraicParameterSet;

    class ByteVolume : public Volume
    {
    public:

        ByteVolume(const Vec3ui& resolution, std::optional<float> initValue, std::optional<float> maximumInputFloatValue = 1.0f);
        ByteVolume(const Vec3ui& resolution, const BoundingBox3f& bbox, std::optional<float> initValue, std::optional<float> maximumInputFloatValue = 1.0f);
        ByteVolume(const Vec3ui& resolution, const byte_t* initialData, std::optional<float> maximumInputFloatValue = 1.0f);
        ~ByteVolume();

        std::unique_ptr<float> convertToFloat() override;

        void setZSlice(unsigned int z, Image* image) override;
        void setVoxelToValue(const size_t index, float value) override;
        float getVoxelValue(size_t index) const override;

		inline byte_t  nativeVoxelValue(size_t index) const { return data[index]; }
		inline byte_t& nativeVoxelValue(size_t index) { return data[index]; }

		void setVoxelToByteValue(size_t index, byte_t value) const;

        byte_t* getBytePointerStartingFrom(size_t offset = 0) const;

    protected:
        void allocateMemory() override;
        void fillMemoryInNativeFormat(void* addr, float value, size_t count) override;
        void* getRawPointerStartingFrom(size_t offset = 0) const override;

        void init(const float* initialData);
        void init(const byte_t* initialData);

    private:
        float convertValueToFloat(byte_t value) const;
        byte_t convertValueFromFloat(float value) const;

    private:
        byte_t *data;
        float maximumInputFloatValue;
    };
};
