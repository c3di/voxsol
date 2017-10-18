#pragma once
#include <vector>
#include "math/Matrix3x3.h"

class RHSMatricesReader {
public:

    std::string findFullPathFromRelativeFilename(const std::string& filename);
    std::vector<Matrix3x3> read(const std::string& filename);

};