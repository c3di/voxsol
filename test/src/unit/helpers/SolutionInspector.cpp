#include "helpers/SolutionInspector.h"
#include "problem/ProblemFragment.h"
#include "problem/DiscreteProblem.h"
#include <sstream>

SolutionInspector::SolutionInspector(DiscreteProblem& problem) :
    Solution(problem)
{

}

ConfigId SolutionInspector::getEquationIdForFragment(ProblemFragment& fragment) {
    unsigned int index = mapToIndex(fragment.getCenterVertex());
    return vertices[index].materialConfigId;
}

bool SolutionInspector::solutionDimensionsMatchProblem(std::string& errMessage) {
    ettention::Vec3ui expectedSize = problem->getSize() + ettention::Vec3ui(1,1,1);

    if (size.x != expectedSize.x || size.y != expectedSize.y || size.z != expectedSize.z) {
        std::stringstream ss;
        ss << "Solution size " << size << " does not match expected size " << expectedSize;
        errMessage = ss.str();
        return false;
    }

    int expectedNumVertices = expectedSize.x*expectedSize.y*expectedSize.z;
    if (vertices.size() != expectedNumVertices) {
        std::stringstream ss;
        ss << "Number of vertices " << vertices.size() << " does not match expected number " << expectedNumVertices;
        errMessage = ss.str();
        return false;
    }

    return true;
}

bool SolutionInspector::matConfigEquationIdsMatchPositionInVector(std::string& errMessage) {
    for (int i = 0; i < matConfigEquations.size(); i++) {
        if (matConfigEquations[i].getId() != i) {
            std::stringstream ss;
            ss << "MaterialConfigEquation at position " << i << " should have matching id " << i << ", instead it has id " << matConfigEquations[i].getId();
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

bool SolutionInspector::allMatConfigEquationsInitialized(std::string& errMessage) {
    for (int i = 0; i < matConfigEquations.size(); i++) {
        MaterialConfigurationEquations eqn = matConfigEquations[i];
        if (eqn.getId() < 0) {
            std::stringstream ss;
            ss << "MaterialConfigurationEquation " << i << " has invalid id " << eqn.getId();
            errMessage = ss.str();
            return false;
        }

        const Matrix3x3* lhs = eqn.getLHSInverse();
        if (lhs == NULL || *lhs == Matrix3x3::identity) {
            std::stringstream ss;
            ss << "MaterialConfigurationEquation " << i << " has invalid LHS matrix";
            errMessage = ss.str();
            return false;
        }

        for (int j = 0; j < 27; j++) {
            const Matrix3x3* rhs = eqn.getRHS(j);
            if (rhs == NULL || *rhs == Matrix3x3::identity) {
                std::stringstream ss;
                ss << "MaterialConfigurationEquation " << i << " has invalid RHS matrix for vertex " << j;
                errMessage = ss.str();
                return false;
            }
        }
    }
    return true;
}

bool SolutionInspector::allVerticesHaveValidSignatureId(std::string& errMessage) {
    for (int i = 0; i < vertices.size(); i++) {
        int equationId = vertices[i].materialConfigId;
        if (equationId < 0 || equationId > matConfigEquations.size()) {
            std::stringstream ss;
            ss << "Vertex " << i << " has invalid material configuration equation Id " << equationId;
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

MaterialConfigurationEquations* SolutionInspector::getEquationsForFragment(ProblemFragment& fragment) {
    Vertex vertex = vertices.at(mapToIndex(fragment.getCenterVertex()));
    return &matConfigEquations.at(vertex.materialConfigId);
}

