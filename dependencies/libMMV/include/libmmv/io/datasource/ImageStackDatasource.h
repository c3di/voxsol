#pragma once

#include "io/HyperStackIndex.h"
#include "model/image/Image.h"

namespace ettention
{
    class ImageStackDataSource
    {
    public:
        ImageStackDataSource();
        ~ImageStackDataSource();

        virtual Vec2ui getResolution() const = 0;
        virtual unsigned int getNumberOfProjections() const = 0;
        virtual Image* loadProjectionImage(const HyperStackIndex& index) = 0;
        virtual const char* getName() const = 0;
        virtual HyperStackIndex firstIndex() const = 0;
        virtual HyperStackIndex lastIndex() const = 0;

    };
}