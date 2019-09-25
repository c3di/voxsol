#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "gpu/sampling/ResidualVolume.h"
#include "solution/Solution.h"

class ResidualVolume;
class SolutionAnalyzer;

class VTKSolutionWriter 
{
public:
    VTKSolutionWriter(Solution* solution, ResidualVolume* importanceVolume = nullptr);
    ~VTKSolutionWriter();

	void virtual writeEntireStructureToFile(const std::string& filename);
	
    void filterOutNullVoxels();
    void setMechanicalValuesOutput(bool flag);

    unsigned int numberOfCells;
    unsigned int numberOfPoints;

protected:
	void fillFilteredPointMap();
	void fillFilteredCellMap();
	unsigned int getMappedIndex( unsigned int originalIndex );
	
    Solution* solution;
    ResidualVolume* importanceVolume;
    bool enableResidualOutput = false;
    bool enableMatConfigIdOutput = true;
    bool enableMechanicalValuesOutput = false;
    

    // For filtered case:
    bool nullVoxelsWereFiltered = false;
    std::unordered_map<unsigned int, unsigned int> pointMapOriginalToFiltered;
    std::unordered_map<unsigned int, unsigned int> pointMapFilteredToOriginal;
    std::unordered_map<unsigned int, unsigned int> cellMapOriginalToFiltered;
	std::unordered_map<unsigned int, unsigned int> cellMapFilteredToOriginal;
    bool isDisplacementBoundary(unsigned int vertexIndex);
    bool vertexWasFilteredOut(unsigned int vertexIndex);

	void virtual writeEntireStructureToStream(std::ostream& stream);
    void virtual writeHeader(std::ostream& stream);
    void writePoints(std::ostream& stream);
	void writeOnePoint(std::ostream& stream, unsigned int originalIndex);
    void writeCells(std::ostream& stream);
    void writeCell(std::ostream& stream, VoxelCoordinate& coord);
    void writeVertexToCell(std::ostream& stream, unsigned int xi, unsigned int yi, unsigned int zi, VoxelCoordinate& coord);
    void writeCellTypes(std::ostream& stream);
    void writeCellData(std::ostream& stream);
    void writePointData(std::ostream& stream);

    // Cell data
    void writeMaterials(std::ostream& stream);
    void writeVonMisesStresses(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer);
    void writeVonMisesStrains(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer);
    void writeStressTensors(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer);
    void writeStrainTensors(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer);

    // Point data
    void writeDisplacements(std::ostream& stream);
    void writeResiduals(std::ostream& stream);
    void writeBoundaries(std::ostream& stream);
    void writeMaterialConfigIds(std::ostream& stream);
};
