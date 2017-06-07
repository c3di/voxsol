#pragma once
#include <stdio.h>

#include "io/serializer/VolumeFileFormatWriter.h"
#include "model/volume/Volume.h"

namespace ettention
{

    class VolumeSerializer
    {
    public:
        VolumeSerializer();
        virtual ~VolumeSerializer();

        void registerVolumeFileFormat(VolumeFileFormatWriter* writer);
        void write(Volume* volume, std::string outputVolumeFileName, VoxelValueType mode, CoordinateOrder order);
        void write(Volume* volume, std::string outputVolumeFileName, std::string specificFileExtension, VoxelValueType mode, CoordinateOrder order);

    protected:
        VolumeFileFormatWriter* findWriter(std::string specifiedExtension, std::string outputVolumeFileName);
        std::unordered_map<std::string, VolumeFileFormatWriter*> fileFormats;
    };

}
