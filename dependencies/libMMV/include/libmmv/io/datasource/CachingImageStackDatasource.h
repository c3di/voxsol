#pragma once
#include <unordered_map>
#include <list>
#include "libmmv/io/datasource/ImageStackDataSource.h"

namespace libmmv
{
    class Image;
    class HyperStackIndex;

    class CachingImageStackDataSource : public ImageStackDataSource
    {
    public:
        static const unsigned int INFINITE_CACHED_IMAGES = (unsigned int)-1;

        CachingImageStackDataSource(unsigned int maxCachedImages = 16);
        virtual ~CachingImageStackDataSource();

        Image* loadProjectionImage(const HyperStackIndex& index) override;
        void cacheProjectionImage(const HyperStackIndex& index, Image* img);
        void clearCachedProjectionImages();
        void setMaxCachedImages(unsigned int max);

    protected:
        unsigned int maxCachedImages;
        typedef std::pair<HyperStackIndex, Image*> LruEntry;
        std::list<LruEntry> lruProjectionImages;
        std::unordered_map<HyperStackIndex, Image*, HyperStackIndexHash> projectionImages;

        void reorganizeCache();
    };
}