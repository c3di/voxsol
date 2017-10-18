#include "stdafx.h"
#include "RHSMatricesReader.h"
#include <iostream>
#include <fstream>
#include <experimental/filesystem>


std::vector<Matrix3x3> RHSMatricesReader::read(const std::string& filename) {
    std::ifstream file;
    std::string fullPath = findFullPathFromRelativeFilename(filename);
    file.open(fullPath, std::ios_base::in);

    if (!file) {
        throw "Could not open test data file " + filename;
    }

    std::vector<Matrix3x3> rhsMatrices;
    std::string line;
    std::getline(file, line); // First line is a comment

    for (int i = 0; i < 27; i++) {
        std::getline(file, line);
        std::istringstream in(line);

        std::vector<REAL> values;
        for (int i = 0; i < 9; i++) {
            REAL val;
            in >> val;
            values.push_back(val);
        }
        Matrix3x3 rhs(values);
        rhsMatrices.push_back(rhs);
    }

    file.close();

    return rhsMatrices;
}

std::string RHSMatricesReader::findFullPathFromRelativeFilename(const std::string& filename) {
    namespace fs = std::tr2::sys;
    // or namespace fs = boost::filesystem;
    auto baseDir = fs::current_path();
    while (baseDir.has_parent_path())
    {
        auto combinePath = baseDir / filename;
        if (fs::exists(combinePath))
        {
            return combinePath.string();
        }
        baseDir = baseDir.parent_path();
    }
    throw std::runtime_error("File not found!");
}

