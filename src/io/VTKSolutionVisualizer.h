#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <solution\Solution.h>


class VTKSolutionVisualizer {
public:
    VTKSolutionVisualizer(Solution* solution);
    ~VTKSolutionVisualizer();

    void writeToFile(const std::string& filename);

    void filterOutNullVoxels(bool doFilter);

    unsigned int numberOfCells;
    unsigned int numberOfVertices;

private:
    Solution* solution;
    std::ofstream outFile;
    

    // For filtered case:
    bool filterNullVoxels = false;
    std::unordered_map<unsigned int, unsigned int> vertexOrigToFilteredIndex;
    std::unordered_map<unsigned int, unsigned int> vertexFilteredToOrigIndex;
    std::unordered_map<unsigned int, unsigned int> cellOrigToFilteredIndex;
    std::unordered_map<unsigned int, unsigned int> cellFilteredToOrigIndex;

    void writeHeader();
    void writePositions();
    void writeCells();
    void writeCell(VoxelCoordinate& coord);
    void writeVertexToCell(unsigned int xi, unsigned int yi, unsigned int zi, VoxelCoordinate& coord);
    void writeCellTypes();
    void writeCellData();
    void writePointData();

    // Cell data
    void writeMaterials();

    // Point data
    void writeDisplacements();
    void writeDeltas();
    void writeBoundaries();
};
