#include "helpers/SolutionInspector.h"
#include <sstream>

SolutionInspector::SolutionInspector(DiscreteProblem& problem) :
    Solution(problem)
{

}

unsigned short SolutionInspector::getSignatureIdForFragment(ProblemFragment& fragment) {
    unsigned int index = mapToIndex(fragment.getCenterVertex());
    return signatureIds[index];
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
    if (signatureIds.size() != expectedNumVertices) {
        std::stringstream ss;
        ss << "Number of vertices " << signatureIds.size() << " does not match expected number " << expectedNumVertices;
        errMessage = ss.str();
        return false;
    }

    return true;
}

bool SolutionInspector::fragmentSignatureIdsMatchPositionInVector(std::string& errMessage) {
    for (int i = 0; i < fragmentSignatures.size(); i++) {
        if (fragmentSignatures[i].getId() != i) {
            std::stringstream ss;
            ss << "FragmentSignature at position " << i << " should have matching id " << i << ", instead it has id " << fragmentSignatures[i].getId();
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

bool SolutionInspector::allFragmentSignaturesInitialized(std::string& errMessage) {
    for (int i = 0; i < fragmentSignatures.size(); i++) {
        FragmentSignature sig = fragmentSignatures[i];
        if (sig.getId() < 0) {
            std::stringstream ss;
            ss << "FragmentSignature " << i << " has invalid id " << sig.getId();
            errMessage = ss.str();
            return false;
        }

        const Matrix3x3* lhs = sig.getLHS();
        if (lhs == NULL || *lhs == Matrix3x3::identity) {
            std::stringstream ss;
            ss << "FragmentSignature " << i << " has invalid LHS matrix";
            errMessage = ss.str();
            return false;
        }

        for (int j = 0; j < 27; j++) {
            const Matrix3x3* rhs = sig.getRHS(j);
            if (rhs == NULL || *rhs == Matrix3x3::identity) {
                std::stringstream ss;
                ss << "FragmentSignature " << i << " has invalid RHS matrix for vertex " << j;
                errMessage = ss.str();
                return false;
            }
        }
    }
    return true;
}

bool SolutionInspector::allVerticesHaveValidSignatureId(std::string& errMessage) {
    for (int i = 0; i < signatureIds.size(); i++) {
        int signatureId = signatureIds[i];
        if (signatureId < 0 || signatureId > fragmentSignatures.size()) {
            std::stringstream ss;
            ss << "Vertex " << i << " has invalid signature Id " << signatureId;
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

