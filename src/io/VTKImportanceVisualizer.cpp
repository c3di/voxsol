#include "stdafx.h"
#include "VTKImportanceVisualizer.h"
#include "cuda_runtime.h"

using namespace std;

VTKImportanceVisualizer::VTKImportanceVisualizer(DiscreteProblem* problem, ResidualVolume* vol) :
    importanceVolume(vol),
    problem(problem)
{

}

VTKImportanceVisualizer::~VTKImportanceVisualizer() {

}

void VTKImportanceVisualizer::writeAllLevels(const string& filePrefix) {
    unsigned int numLevels = importanceVolume->getNumberOfLevels();
    for (unsigned int i = 0; i < numLevels; i++) {
        stringstream ss;
        ss << filePrefix << "_" << i << ".vtk";
        writeToFile(ss.str(), i);
    }
}

void VTKImportanceVisualizer::writeToFile(const string& filename, unsigned int level) {
    outFile.open(filename, ios::out);

    libmmv::Vec3<REAL> voxelSize = problem->getVoxelSize();
    voxelSize = voxelSize * asREAL(2.0);
    for (unsigned int i = 0; i < level; i++) {
        voxelSize = voxelSize * asREAL(2.0);
    }
    LevelStats* levelStats = importanceVolume->getPointerToStatsForLevel(level);
    libmmv::Vec3ui levelSize(levelStats->sizeX, levelStats->sizeY, levelStats->sizeZ);

    cout << "Level " << level << " with size " << levelSize << " start index " << levelStats->startIndex << endl;

    writeHeader();
    writePositions(level, voxelSize, levelSize);
    writeResiduals(level, levelSize);

    outFile.close();
}

void VTKImportanceVisualizer::writeHeader() {
    outFile << "# vtk DataFile Version 2.0" << endl;
    outFile << "Stochastic Mechanical Solver: Importance Volume Debug Output" << endl;
    outFile << "ASCII" << endl;
    outFile << "DATASET UNSTRUCTURED_GRID" << endl << endl;
}

void VTKImportanceVisualizer::writePositions(unsigned int level, libmmv::Vec3<REAL>& voxelSize, libmmv::Vec3ui& levelSize) {
    unsigned int numVertices = levelSize.x * levelSize.y * levelSize.z;
    outFile << "POINTS " << numVertices << " float" << endl;

    for (unsigned int z = 0; z < levelSize.z; z++) 
        for (unsigned int y = 0; y < levelSize.y; y++)
            for (unsigned int x = 0; x < levelSize.x; x++) {
                outFile << x * voxelSize.x << " " << y * voxelSize.y << " " << z * voxelSize.z << " " << endl;
            }

    outFile << endl;
}

void VTKImportanceVisualizer::writeResiduals(unsigned int level, libmmv::Vec3ui& levelSize) {
    unsigned int numVertices = levelSize.x * levelSize.y * levelSize.z;
    outFile << "POINT_DATA " << numVertices << endl;
    outFile << "VECTORS residual float" << endl;

    REAL* residual = importanceVolume->getPointerToLevel(level);
    for (unsigned int i = 0; i < numVertices; i++, residual++) {
        outFile << *residual << " " << *residual << " " << *residual << endl;
    }

    outFile << endl;
}


