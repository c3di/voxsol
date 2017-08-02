#pragma once
#include "solution/Solution.h"

class SolutionInspector : public Solution {
public:
    SolutionInspector(DiscreteProblem& problem);

    bool solutionDimensionsMatchProblem(std::string& errMessage);
    bool matrixStoreIdsMatchPositionInVector(std::string& errMessage);
    bool allMatrixStoresInitialized(std::string& errMessage);
    bool allVerticesHaveValidSignatureId(std::string& errMessage);
};

