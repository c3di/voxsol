#pragma once
#include "solution/Solution.h"

class SolutionInspector : public Solution {
public:
    SolutionInspector(DiscreteProblem& problem);

    bool solutionDimensionsMatchProblem(std::string& errMessage);
    bool fragmentSignatureIdsMatchPositionInVector(std::string& errMessage);
    bool allFragmentSignaturesInitialized(std::string& errMessage);
    bool allVerticesHaveValidSignatureId(std::string& errMessage);
};

