#pragma once

#include <fstream>
#include <iostream>
#include <string>

class ResidualVolume;
class Solution;
class SolutionAnalyzer;

class VTKSolutionWriter 
{
public:
    VTKSolutionWriter(Solution* solution, ResidualVolume* importanceVolume = nullptr);
    ~VTKSolutionWriter();

    void writeEntireStructureToFile(const std::string& filename);
	void writeEntireStructureToStream(std::ostream& stream);
	
    void filterOutNullVoxels();
    void setMechanicalValuesOutput(bool flag);

    unsigned int numberOfCells;
    unsigned int numberOfPoints;

private:
	void fillFilteredPointMap();
	void fillFilteredCellMap();
	unsigned int getMappedIndex( unsigned int originalIndex );
	
private:
    Solution* solution;
    ResidualVolume* importanceVolume;
    bool enableResidualOutput = false;
    bool enableMatConfigIdOutput = false;
    bool enableMechanicalValuesOutput = false;
    

    // For filtered case:
    bool nullVoxelsWereFiltered = false;
    std::unordered_map<unsigned int, unsigned int> pointMapOriginalToFiltered;
    std::unordered_map<unsigned int, unsigned int> pointMapFilteredToOriginal;
    std::unordered_map<unsigned int, unsigned int> cellMapOriginalToFiltered;
	std::unordered_map<unsigned int, unsigned int> cellMapFilteredToOriginal;

    void writeHeader(std::ostream& stream);
    void writePoints(std::ostream& stream);
	void writeOnePoint(std::ostream& stream, unsigned int originalIndex);
    void writeCells(std::ostream& stream);
    void writeCell(std::ostream& stream, VoxelCoordinate& coord);
    void writeVertexToCell(std::ostream& stream, unsigned int xi, unsigned int yi, unsigned int zi, VoxelCoordinate& coord);
    void writeCellTypes(std::ostream& stream);
    void writeCellData(std::ostream& stream);
    void writePointData(std::ostream& stream);

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
