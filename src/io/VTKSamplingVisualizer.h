#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include "solution/Solution.h"
#include "cuda_runtime.h"

class VTKSamplingVisualizer {
public:
    VTKSamplingVisualizer(Solution* solution);
    ~VTKSamplingVisualizer();

    void writeToFile(const std::string& filename, const uint3* blockOrigins, int numBlocks, int blockSize);

private:
    Solution* solution;
    std::ofstream outFile;

    void writeHeader();
    void writePositions(const uint3* blockOrigins, int numBlocks, int blockSize);
    //void writeActiveRegions(const int3* blockOrigins, int numBlocks, int blockSize);
};