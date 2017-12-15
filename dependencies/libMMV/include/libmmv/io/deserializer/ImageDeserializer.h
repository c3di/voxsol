#pragma once
#include <string>

class fipImage;

namespace libmmv
{
    class Image;

    class ImageDeserializer
    {
    public:
        static Image* readImage(std::string filenameTrunk);

        static unsigned int getNumberOfImagesInStack(std::string filename);
        static Image* readImageFromStack(std::string filename, unsigned int index);

    protected:
        static Image* convertFipImage(fipImage& fimg, std::string filename);

        ImageDeserializer();
        virtual ~ImageDeserializer();

    };
}