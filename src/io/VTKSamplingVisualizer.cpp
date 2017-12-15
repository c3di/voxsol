#include "stdafx.h"
#include "VTKSamplingVisualizer.h"
#include "cuda_runtime.h"

using namespace std;

VTKSamplingVisualizer::VTKSamplingVisualizer(Solution* sol) :
    solution(sol)
{

}

VTKSamplingVisualizer::~VTKSamplingVisualizer() {

}

void VTKSamplingVisualizer::writeToFile(const string& filename, const int3* blockOrigins, int numBlocks, int blockSize) {
    outFile.open(filename, ios::out);

    writeHeader();
    writePositions(blockOrigins, numBlocks, blockSize);
    //writeActiveRegions(blockOrigins, numBlocks, blockSize);

    outFile.close();
}

void VTKSamplingVisualizer::writeHeader() {
    outFile << "# vtk DataFile Version 2.0" << endl;
    outFile << "Stochastic Mechanical Solver: Sampling Debug Output" << endl;
    outFile << "ASCII" << endl;
    outFile << "DATASET UNSTRUCTURED_GRID" << endl << endl;
}

void VTKSamplingVisualizer::writePositions(const int3* blockOrigins, int numBlocks, int blockSize) {
    int numVerticesPerBlock = blockSize*blockSize*blockSize;
    int numVertices = numVerticesPerBlock * numBlocks;
    libmmv::Vec3<REAL> voxelSize = solution->getProblem()->getVoxelSize();
    
    outFile << "POINTS " << numVertices << " float" << endl;
#pragma warning( push )
#pragma warning( disable : 4018)
    for (unsigned int block = 0; block < numBlocks; block++) {
        int blockOriginX = blockOrigins[block].x + 1; //+1 because block origins include the 1-vertex fixed border
        int blockOriginY = blockOrigins[block].y + 1;
        int blockOriginZ = blockOrigins[block].z + 1;

        for (int vz = 0; vz < blockSize; vz++) {
            for (int vy = 0; vy < blockSize; vy++) {
                for (int vx = 0; vx < blockSize; vx++) {
                    REAL x = (blockOriginX + vx) * voxelSize.x;
                    REAL y = (blockOriginY + vy) * voxelSize.y;
                    REAL z = (blockOriginZ + vz) * voxelSize.z;

                    outFile << x << " " << y << " " << z << " " << endl;
                }
            }
        }
    }
#pragma warning(pop)

    outFile << endl;
}



