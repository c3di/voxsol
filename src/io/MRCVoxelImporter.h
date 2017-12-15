#pragma once
#include <string>
#include "libmmv/io/datasource/MRCDataSource.h"
#include "libmmv/math/Vec3.h"

class MaterialDictionary;
class DiscreteProblem;

class MRCVoxelImporter : public libmmv::MRCDataSource {
public:
    MRCVoxelImporter();
    MRCVoxelImporter(std::string& filepath);
    ~MRCVoxelImporter();

    libmmv::Vec3<REAL> getVoxelSizeInMeters() const;
    libmmv::Vec3ui getDimensionsInVoxels() const;

    DiscreteProblem extractDiscreteProblem(MaterialDictionary* materialDictionary);
    MaterialDictionary extractMaterialDictionary();

private:
    unsigned char boneMatId = 0;
    unsigned char implantMatId = 0;

    int nullMaterialColorValue = 0;
    int boneMaterialColorValue = 0;
    int implantMaterialColorValue = 0;


    void addVoxelSliceToProblem(unsigned int zLayer, DiscreteProblem* problem, libmmv::Image* image, libmmv::Vec3ui* stats);
    void readMaterialClassesFromHeader();
};

