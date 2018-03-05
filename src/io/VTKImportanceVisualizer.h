#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include "gpu/sampling/ResidualVolume.h"
#include "cuda_runtime.h"

class VTKImportanceVisualizer {
public:
    VTKImportanceVisualizer(DiscreteProblem* problem, ResidualVolume* importanceVolume);
    ~VTKImportanceVisualizer();

    void writeToFile(const std::string& filename, unsigned int level);
    void writeAllLevels(const std::string& filePrefix);

private:
    ResidualVolume* importanceVolume;
    DiscreteProblem* problem;
    std::ofstream outFile;

    void writeHeader();
    void writePositions(unsigned int level, libmmv::Vec3<REAL>& voxelSize, libmmv::Vec3ui& levelSize);
    void writeResiduals(unsigned int level, libmmv::Vec3ui& levelSize);
};