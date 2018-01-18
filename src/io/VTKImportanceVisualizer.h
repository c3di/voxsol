#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include "gpu/sampling/ImportanceVolume.h"
#include "cuda_runtime.h"

class VTKImportanceVisualizer {
public:
    VTKImportanceVisualizer(DiscreteProblem* problem, ImportanceVolume* importanceVolume);
    ~VTKImportanceVisualizer();

    void writeToFile(const std::string& filename, unsigned int level);
    void writeAllLevels(const std::string& filePrefix);

private:
    ImportanceVolume* importanceVolume;
    DiscreteProblem* problem;
    std::ofstream outFile;

    void writeHeader();
    void writePositions(unsigned int level, libmmv::Vec3<REAL>& voxelSize, libmmv::Vec3ui& levelSize);
    void writeResiduals(unsigned int level, libmmv::Vec3ui& levelSize);
};