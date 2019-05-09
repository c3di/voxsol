#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <solution\Solution.h>
#include <gpu/sampling/ResidualVolume.h>

class SolutionAnalyzer;

class VTKSolutionVisualizer {
public:
    VTKSolutionVisualizer(Solution* solution, ResidualVolume* impVol = nullptr);
    ~VTKSolutionVisualizer();

    void writeToFile(const std::string& filename);
    void writeOnlyDisplacements(const std::string& filename);
    void filterOutNullVoxels(bool doFilter);
    void setMechanicalValuesOutput(bool flag);

    unsigned int numberOfCells;
    unsigned int numberOfVertices;

private:
    Solution* solution;
    ResidualVolume* impVol;
    std::ofstream outFile;
    bool enableResidualOutput = true;
    bool enableMatConfigIdOutput = true;
    bool enableMechanicalValuesOutput = false;
    

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
    void writeVonMisesStresses(SolutionAnalyzer* solutionAnalyzer);
    void writeVonMisesStrains(SolutionAnalyzer* solutionAnalyzer);
    void writeStressTensors(SolutionAnalyzer* solutionAnalyzer);
    void writeStrainTensors(SolutionAnalyzer* solutionAnalyzer);

    // Point data
    void writeDisplacements();
    void writeResiduals();
    void writeBoundaries();
    void writeMaterialConfigIds();
};
