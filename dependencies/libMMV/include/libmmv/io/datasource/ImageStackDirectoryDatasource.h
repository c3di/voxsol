#pragma once
#include <vector>
#include <map>
#include "libmmv/io/datasource/CachingImageStackDataSource.h"

namespace libmmv
{
    class ImageStackDirectoryDataSource : public CachingImageStackDataSource
    {
    protected:
        class ImageLocation
        {
        protected:
            std::string path;
            unsigned int indexInImageStack;

        public:
            static const unsigned int NOT_INSIDE_IMAGE_STACK = (unsigned int)-1;

            ImageLocation();
            ImageLocation(const std::string& path, unsigned int indexInImageStack = NOT_INSIDE_IMAGE_STACK);

            const std::string& getPath() const;
            bool isInsideImageStack() const;
            unsigned int getIndexInImageStack() const;
        };

    public:
        ImageStackDirectoryDataSource();
        ~ImageStackDirectoryDataSource();

        std::vector<HyperStackIndex> collectAllValidIndices() const;

        Vec2ui getResolution() const override;
        const char* getName() const override;
        unsigned int getNumberOfProjections() const override;
        Image* loadProjectionImage(const HyperStackIndex& index) override;
        virtual HyperStackIndex firstIndex() const override;
        virtual HyperStackIndex lastIndex() const override;

    protected:
        std::string directory;
        Vec2ui resolution;
        std::map<HyperStackIndex, ImageLocation> imageLocations;

        std::string getAbsoluteImageLocation(const std::string& location);
        Image* loadImageFromLocation(const ImageLocation& location);
    };
}