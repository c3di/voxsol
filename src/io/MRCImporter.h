#pragma once
#include <string>
#include "libmmv/io/datasource/MRCDataSource.h"
#include "libmmv/math/Vec3.h"
#include "material/MaterialDictionary.h"

class DiscreteProblem;

class MRCImporter : public libmmv::MRCDataSource {
public:
    MRCImporter();
    MRCImporter(std::string& filepath);
    ~MRCImporter();

    libmmv::Vec3<REAL> getVoxelSizeInMeters() const;
    libmmv::Vec3ui getDimensionsInVoxels() const;

    void populateDiscreteProblem(DiscreteProblem* problem);

    void addMaterialMapping(Material* mat, unsigned char colorVal);

private:
    std::unordered_map<unsigned char, Material*> materialMap;

    void addVoxelSliceToProblem(unsigned int zLayer, DiscreteProblem* problem, libmmv::Image* image, int* stats);

};

