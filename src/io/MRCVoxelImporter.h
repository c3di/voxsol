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

    void populateDiscreteProblem(DiscreteProblem* problem);
    MaterialDictionary extractMaterialDictionary();
    void setVoxelSizeInMeters(libmmv::Vec3<REAL>& voxelSize);

    int nullMaterialColorValue = 0;
    int boneMaterialColorValue = 0;
    int implantMaterialColorValue = 0;

    int linerMaterialColorValue = 0;
    int fatMaterialColorValue = 0;
    int muscleMaterialColorValue = 0;
    int skinMaterialColorValue = 0;
    int socketMaterialColorValue = 0;

    unsigned char boneMatId = 0;
    unsigned char implantMatId = 0;

    unsigned char linerMatId = 0;
    unsigned char fatMatId = 0;
    unsigned char socketMatId = 0;
    unsigned char muscleMatId = 0;

private:

    void addVoxelSliceToProblem(unsigned int zLayer, DiscreteProblem* problem, libmmv::Image* image, int* stats);
    void readMaterialClassesFromHeader();
};

