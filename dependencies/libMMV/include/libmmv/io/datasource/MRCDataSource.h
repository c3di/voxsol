#pragma once

#include <boost/filesystem.hpp>

#include "math/vec2.h"
#include "io/datasource/MRCHeader.h"
#include "io/datasource/CachingImageStackDataSource.h"

namespace ettention
{
    class MRCDataSource : public CachingImageStackDataSource
    {

    public:
        MRCDataSource();
        MRCDataSource(const boost::filesystem::path& stackFilePath, bool logaritmizeData);
        MRCDataSource(const boost::filesystem::path& stackFilePath, const boost::filesystem::path& tiltFilePath, bool logaritmizeData);
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
        boost::filesystem::path stackFilePath;
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