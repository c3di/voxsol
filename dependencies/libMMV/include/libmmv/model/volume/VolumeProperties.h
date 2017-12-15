#pragma once

#include <vector>
#include "libmmv/model/volume/BoundingBox.h"
#include "libmmv/math/Vec3.h"
#include "libmmv/model/volume/Voxel.h"

namespace libmmv
{
    class VolumeProperties
    {
    public:
        VolumeProperties(Voxel::DataType voxelType, const Vec3ui& resolution);
        VolumeProperties(Voxel::DataType voxelType, const Vec3ui& resolution, const BoundingBox3f& bbox);
        VolumeProperties(Voxel::DataType voxelType, const Vec3ui& resolution, const Vec3f& voxelSize);
        ~VolumeProperties();

        static const unsigned int SUBVOLUME_COUNT_AUTO = 0U;

        void init();

        Voxel::DataType getVoxelType() const;
        size_t getVoxelByteWidth() const;
        const Vec3f& getVoxelSize() const;

        const Vec3ui& getVolumeResolution() const;
        const BoundingBox3f& getRealSpaceBoundingBox() const;
		const BoundingBox3i getMemoryBoundingBox() const;
        size_t getVolumeVoxelCount() const;
        size_t getVolumeByteWidth() const;

        unsigned int getSubVolumeCount() const;
        void setSubVolumeCount(unsigned int count);

        const Vec3ui& getSubVolumeResolution(unsigned int index) const;
        const Vec3ui& getSubVolumeBaseCoords(unsigned int index) const;
        const BoundingBox3f& getSubVolumeBoundingBox(unsigned int index) const;
        size_t getSubVolumeVoxelCount(unsigned int index) const;
        size_t getSubVolumeByteWidth(unsigned int index) const;

    private:
        Voxel::DataType voxelType;
        Vec3f voxelSize;
        Vec3ui resolution;
        Vec3f origin;
        size_t voxelCount;
        size_t sizeInBytes;
        BoundingBox3f boundingBox;
        std::vector<Vec3ui> subVolumeResolutions;
        std::vector<Vec3ui> subVolumeBaseCoords;
        std::vector<BoundingBox3f> subVolumeBoundingBoxes;
    };
}
