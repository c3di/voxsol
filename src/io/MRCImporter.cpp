#include "stdafx.h"
#include "MRCImporter.h"
#include "problem/DiscreteProblem.h"

MRCImporter::MRCImporter()
{
}

MRCImporter::MRCImporter(std::string & filepath) : 
    libmmv::MRCDataSource(filepath, false)
{
}

MRCImporter::~MRCImporter()
{
}

libmmv::Vec3<REAL> MRCImporter::getVoxelSizeInMeters() const
{
    libmmv::Vec3<REAL> voxelSize(asREAL(0.0046), asREAL(0.0046), asREAL(0.0046));
    return voxelSize;
}

libmmv::Vec3ui MRCImporter::getDimensionsInVoxels() const
{
    return libmmv::Vec3ui(mrcHeader.nx, mrcHeader.ny, mrcHeader.nz);
}

void MRCImporter::populateDiscreteProblem(DiscreteProblem * problem)
{
    std::cout << "Processing " << problem->getSize().x << "x" << problem->getSize().y << "x" << problem->getSize().z << " = " << problem->getNumberOfVoxels() << " voxels...\n";
    int* stats = new int[256];
    for (int i = 0; i <= 255; i++) {
        stats[i] = 0;
    }

    for (unsigned int i = 0; i < numberOfProjections; i++) {
        libmmv::Image* image = loadProjectionImage(i);
        addVoxelSliceToProblem(i, problem, image, stats);
    }
    std::cout << "Assigned materials: ";
    for (int i = 0; i <= 255; i++) {
        if (stats[i] > 0) {
            std::cout << i << ": " << stats[i] << " voxels, ";
        }
    }
    std::cout << std::endl;
    delete[] stats;
}

void MRCImporter::addMaterialMapping(Material * mat, unsigned char colorVal)
{
    materialMap.emplace(colorVal, mat);
}

void MRCImporter::addVoxelSliceToProblem(unsigned int zLayer, DiscreteProblem * problem, libmmv::Image * image, int * stats)
{
    libmmv::Vec3ui coord(0, 0, zLayer);

    for (coord.x = 0; coord.x < image->getResolution().x; coord.x++) {
        for (coord.y = 0; coord.y < image->getResolution().y; coord.y++) {
            unsigned char colorVal = (unsigned char)image->getPixel(coord.x, coord.y);
        
            if (materialMap.count(colorVal) <= 0) {
                std::stringstream ss;
                ss << "Unmapped color value encountered while importing MRC stack: " << static_cast<unsigned int>(colorVal) << std::endl;
                throw std::ios_base::failure(ss.str().c_str());
            }
            Material* matchedMat = materialMap.at(colorVal);
            stats[colorVal]++;
            problem->setMaterial(coord, matchedMat->id);
        }
    }
}
