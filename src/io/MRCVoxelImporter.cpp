#include "stdafx.h"
#include "MRCVoxelImporter.h"
#include "problem/DiscreteProblem.h"
#include "material/MaterialDictionary.h"
#include "material/MaterialFactory.h"

MRCVoxelImporter::MRCVoxelImporter()
{
}

MRCVoxelImporter::MRCVoxelImporter(std::string & filepath) :
    libmmv::MRCDataSource(filepath, false)
{
    readMaterialClassesFromHeader();
}

MRCVoxelImporter::~MRCVoxelImporter()
{
}

libmmv::Vec3<REAL> MRCVoxelImporter::getVoxelSizeInMeters() const 
{
    // MRC voxel (cell) sizes are in mm
    //libmmv::Vec3<REAL> voxelSize(mrcHeader.cellDimX / asREAL(1000.0), mrcHeader.cellDimY / asREAL(1000.0), mrcHeader.cellDimZ / asREAL(1000.0));
    //libmmv::Vec3<REAL> voxelSize(asREAL(0.0024), asREAL(0.0024), asREAL(0.0030)); //stumpf
    //libmmv::Vec3<REAL> voxelSize(asREAL(0.0024), asREAL(0.0024), asREAL(0.0014)); //stumpf
    libmmv::Vec3<REAL> voxelSize(asREAL(0.0046), asREAL(0.0046), asREAL(0.046));
    return voxelSize;
}

libmmv::Vec3ui MRCVoxelImporter::getDimensionsInVoxels() const
{
    return libmmv::Vec3ui(mrcHeader.nx, mrcHeader.ny, mrcHeader.nz);
}

void MRCVoxelImporter::populateDiscreteProblem(DiscreteProblem* problem) {
    std::cout << "Processing " << problem->getSize().x << "x" << problem->getSize().y << "x" << problem->getSize().z << " = " << problem->getNumberOfVoxels() << " voxels...\n";
    int* stats = new int[128];
    for (int i = 0; i < 128; i++) {
        stats[i] = 0;
    }

    for (unsigned int i = 0; i < numberOfProjections; i++) {
        libmmv::Image* image = loadProjectionImage(i);
        addVoxelSliceToProblem(i, problem, image, stats);
    }
    std::cout << "Assigned materials: ";
    for (int i = 0; i < 128; i++) {
        if (stats[i] > 0) {
            std::cout << i << ": " << stats[i] << " voxels, ";
        }
    }

    delete[] stats;
}

MaterialDictionary MRCVoxelImporter::extractMaterialDictionary()
{
    MaterialDictionary dict;
    MaterialFactory mFactory;

    Material boneMat = mFactory.createMaterialWithProperties(asREAL(102000000000), asREAL(0.4));
    Material implantMat = mFactory.createMaterialWithProperties(asREAL(210000000000), asREAL(0.3));
    Material fatMat = mFactory.createMaterialWithProperties(asREAL(800), asREAL(0.4));
    Material muscleMat = mFactory.createMaterialWithProperties(asREAL(1800), asREAL(0.3));
    Material linerMat = mFactory.createMaterialWithProperties(asREAL(1000), asREAL(0.3));
    Material socketMat = mFactory.createMaterialWithProperties(asREAL(112000000000), asREAL(0.3));

    dict.addMaterial(boneMat);
    dict.addMaterial(implantMat);
    dict.addMaterial(fatMat);
    dict.addMaterial(muscleMat);
    dict.addMaterial(linerMat);
    dict.addMaterial(socketMat);

    boneMatId = boneMat.id;
    implantMatId = implantMat.id;
    fatMatId = fatMat.id;
    muscleMatId = muscleMat.id;
    linerMatId = linerMat.id;
    socketMatId = socketMat.id;

    std::cout << "Bone material id: " << (int)boneMatId << std::endl;

    return dict;
}

void MRCVoxelImporter::addVoxelSliceToProblem(unsigned int zLayer, DiscreteProblem* problem, libmmv::Image* image, int* stats)
{
    libmmv::Vec3ui coord(0, 0, zLayer);
    for (coord.x = 0; coord.x < image->getResolution().x; coord.x++) {
        for (coord.y = 0; coord.y < image->getResolution().y; coord.y++) {
            int colorVal = (int)image->getPixel(coord.x, coord.y);

            if (colorVal == nullMaterialColorValue) {
                stats[0]++;
                problem->setMaterial(coord, Material::EMPTY.id);
            }
            else if (colorVal == boneMaterialColorValue) {
                stats[1]++;
                problem->setMaterial(coord, boneMatId);
            }
            else if (colorVal == implantMaterialColorValue) {
                stats[2]++;
                problem->setMaterial(coord, implantMatId);
            }
            else if (colorVal == fatMaterialColorValue || colorVal == skinMaterialColorValue) {
                stats[3]++;
                problem->setMaterial(coord, fatMatId);
            }
            else if (colorVal == muscleMaterialColorValue) {
                stats[4]++;
                problem->setMaterial(coord, muscleMatId);
            }
            else if (colorVal == linerMaterialColorValue) {
                stats[5]++;
                problem->setMaterial(coord, linerMatId);
            }
            else if (colorVal == socketMaterialColorValue) {
                stats[6]++;
                problem->setMaterial(coord, socketMatId);
            }
            else {
                std::stringstream ss;
                ss << "Unmapped color value encountered while importing MRC stack: " << static_cast<unsigned int>(colorVal) << std::endl;
                throw std::ios_base::failure(ss.str().c_str());
            }
            
        }
    }
}

void MRCVoxelImporter::readMaterialClassesFromHeader() {
    nullMaterialColorValue = (int)mrcHeader.dMin;
    implantMaterialColorValue = (int)mrcHeader.dMax;
    boneMaterialColorValue = int(mrcHeader.dMax - mrcHeader.dMin) / 2;
}
