#pragma once
#include "solution/Solution.h"

class SolutionInspector : public Solution {
public:
    SolutionInspector(DiscreteProblem& problem);

    unsigned short getEquationIdForFragment(ProblemFragment& fragment);

    bool solutionDimensionsMatchProblem(std::string& errMessage);
    bool matConfigEquationIdsMatchPositionInVector(std::string& errMessage);
    bool allMatConfigEquationsInitialized(std::string& errMessage);
    bool allVerticesHaveValidSignatureId(std::string& errMessage);
};

