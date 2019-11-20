#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "solution/Solution.h"

class SolutionAnalyzer;

class CSVSolutionWriter {
public:
    CSVSolutionWriter(Solution* solution);
    ~CSVSolutionWriter();

    void writeSolutionToFile(const std::string& filepath);

protected:
    Solution* solution;

    void writeSolutionToStream(std::ostream& stream);
    void writeStrainTensorsForVoxel(std::ostream& stream, const VoxelCoordinate& coord, SolutionAnalyzer& solutionAnalyzer);
    void writeStressTensorsForVoxel(std::ostream& stream, const VoxelCoordinate& coord, SolutionAnalyzer& solutionAnalyzer);
};