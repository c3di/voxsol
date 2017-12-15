#pragma once
#include "libmmv/io/datasource/ImageStackDataSource.h"
#include <string>

namespace libmmv
{
    class Volume;

    class StackComparator
    {
    public:
        StackComparator(ImageStackDataSource& firstInputStack, ImageStackDataSource& secondInputStack);
        ~StackComparator();

        unsigned int getSliceIndexWithMaxRMS();

        float getMaxRMS();

        float getRMSOfSlice(unsigned int sliceIndex);

        static float getRMSBetweenVolumes(std::string firstVolume, std::string secondVolume);
        static void assertVolumesAreEqual(std::string firstVolume, std::string secondVolume);

        static float getRMSBetweenVolumes(Volume* firstVolume, Volume* secondVolume);
        static void assertVolumesAreEqual(Volume* firstVolume, Volume* secondVolume);

    private:
        ImageStackDataSource& firstStack;
        ImageStackDataSource& secondStack;
        static const float acceptableRMSError;

    };
}

