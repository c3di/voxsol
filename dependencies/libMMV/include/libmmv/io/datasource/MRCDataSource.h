#pragma once
#include <fstream>
#include "libmmv/math/vec2.h"
#include "libmmv/io/datasource/MRCHeader.h"
#include "libmmv/io/datasource/CachingImageStackDataSource.h"

namespace libmmv
{
    class MRCDataSource : public CachingImageStackDataSource
    {

    public:
        MRCDataSource();
        MRCDataSource(const std::string& stackFilePath, bool logaritmizeData);
        MRCDataSource(const std::string& stackFilePath, const std::string& tiltFilePath, bool logaritmizeData);
        ~MRCDataSource();

        Vec2ui getResolution() const override;
        unsigned int getNumberOfProjections() const override;
        const char* getName() const override;
        HyperStackIndex firstIndex() const override;
        HyperStackIndex lastIndex() const override;

        Image* loadProjectionImage(const HyperStackIndex& index) override;

    protected:
        std::ifstream file;
        float dataMin;
        float dataMax;
        bool logaritmizeData;
        std::string stackFilePath;
        MRCHeader mrcHeader;
        Vec2ui resolution;
        unsigned int numberOfProjections;

        void init();   
        void readImageData(float* dest, unsigned int projectionIndex);
        
        template<typename _T>
        void readSlice(float* dest, unsigned int projectionIndex, float minValue);

        template<typename _T>
        float convertValue(_T rawValue, float minValue) const;

        template<typename _T>
        void getDataRange();
    };
}