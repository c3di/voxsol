#pragma once
#include "libmmv/io/serializer/VolumeFileFormatWriter.h"
#include <fstream>

namespace libmmv
{
    class MRCWriter : public VolumeFileFormatWriter
    {
    public:
        MRCWriter();
        virtual ~MRCWriter();

        virtual bool canGenerateVoxelType(VoxelValueType mode) override;
        virtual bool canGenerateRotation(CoordinateOrder rotation) override;
        virtual std::string getFileExtension() override;

    protected:
        virtual void writeHeader(std::ofstream &outfile, VolumeProperties& volumeProperties, const Vec3ui volumeResolution, RangeTransformation* rangeTransformation) override;
    };
}