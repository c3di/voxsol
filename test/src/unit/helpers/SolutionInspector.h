#pragma once
#include "solution/Solution.h"

class ProblemFragment;

class SolutionInspector : public Solution {
public:
    SolutionInspector(DiscreteProblem& problem);

    ConfigId getEquationIdForFragment(ProblemFragment& fragment);
    MaterialConfigurationEquations* getEquationsForFragment(ProblemFragment& fragment);

    bool solutionDimensionsMatchProblem(std::string& errMessage);
    bool matConfigEquationIdsMatchPositionInVector(std::string& errMessage);
    bool allMatConfigEquationsInitialized(std::string& errMessage);
    bool allVerticesHaveValidSignatureId(std::string& errMessage);
};

