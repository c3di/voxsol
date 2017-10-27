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

private:
    Solution* solution;
    std::ofstream outFile;
    const unsigned int numberOfCells;
    const unsigned int numberOfVertices;

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
