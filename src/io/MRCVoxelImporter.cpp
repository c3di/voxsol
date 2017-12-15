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
    libmmv::Vec3<REAL> voxelSize(asREAL(0.0024), asREAL(0.0024), asREAL(0.017));
    return voxelSize;
}

libmmv::Vec3ui MRCVoxelImporter::getDimensionsInVoxels() const
{
    return libmmv::Vec3ui(mrcHeader.nx, mrcHeader.ny, mrcHeader.nz);
}

DiscreteProblem MRCVoxelImporter::extractDiscreteProblem(MaterialDictionary* materialDictionary) 
{
    DiscreteProblem problem(getDimensionsInVoxels(), getVoxelSizeInMeters(), materialDictionary);
    std::cout << "Processing " << problem.getSize().x << "x" << problem.getSize().y << "x" << problem.getSize().z << " = "<<problem.getNumberOfVoxels() << " voxels...\n";
    libmmv::Vec3ui stats(0, 0, 0);
    for (unsigned int i = 0; i < numberOfProjections; i++) {
        libmmv::Image* image = loadProjectionImage(i);
        addVoxelSliceToProblem(i, &problem, image, &stats);
    }
    std::cout << "Assigned materials: " << stats.x << " null voxels, " << stats.y << " bone voxels, " << stats.z << " implant voxels\n";
    return problem;
}

MaterialDictionary MRCVoxelImporter::extractMaterialDictionary()
{
    MaterialDictionary dict;
    MaterialFactory mFactory;

    Material boneMat = mFactory.createMaterialWithProperties(asREAL(21000000000), asREAL(0.4));
    Material implantMat = mFactory.createMaterialWithProperties(asREAL(102000000000), asREAL(0.3));

    dict.addMaterial(boneMat);
    dict.addMaterial(implantMat);

    boneMatId = boneMat.id;
    implantMatId = implantMat.id;

    return dict;
}

void MRCVoxelImporter::addVoxelSliceToProblem(unsigned int zLayer, DiscreteProblem* problem, libmmv::Image* image, libmmv::Vec3ui* stats)
{
    libmmv::Vec3ui coord(0, 0, zLayer);
    for (coord.x = 0; coord.x < image->getResolution().x; coord.x++) {
        for (coord.y = 0; coord.y < image->getResolution().y; coord.y++) {
            int colorVal = (int)image->getPixel(coord.x, coord.y);
            if (colorVal == 0) {
                // empty (null) material
                stats->x++;
                problem->setMaterial(coord, 0);
            }
            else if (colorVal == 2) {
                // implant material
                stats->z++;
                problem->setMaterial(coord, implantMatId);
            }
            else {
                // bone material
                stats->y++;
                problem->setMaterial(coord, boneMatId);
            }
        }
    }
}

void MRCVoxelImporter::readMaterialClassesFromHeader() {
    nullMaterialColorValue = (int)mrcHeader.dMin;
    implantMaterialColorValue = (int)mrcHeader.dMax;
    boneMaterialColorValue = int(mrcHeader.dMax - mrcHeader.dMin) / 2;
}
