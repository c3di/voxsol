#include "helpers/SolutionInspector.h"
#include <sstream>

SolutionInspector::SolutionInspector(DiscreteProblem& problem) :
    Solution(problem)
{

}

unsigned short SolutionInspector::getSignatureIdForFragment(ProblemFragment& fragment) {
    unsigned int index = mapToIndex(fragment.getCenterVertex());
    return m_signatureIds[index];
}

bool SolutionInspector::solutionDimensionsMatchProblem(std::string& errMessage) {
    ettention::Vec3ui expectedSize = m_problem->getSize() + ettention::Vec3ui(1,1,1);

    if (m_size.x != expectedSize.x || m_size.y != expectedSize.y || m_size.z != expectedSize.z) {
        std::stringstream ss;
        ss << "Solution size " << m_size << " does not match expected size " << expectedSize;
        errMessage = ss.str();
        return false;
    }

    int expectedNumVertices = expectedSize.x*expectedSize.y*expectedSize.z;
    if (m_signatureIds.size() != expectedNumVertices) {
        std::stringstream ss;
        ss << "Number of vertices " << m_signatureIds.size() << " does not match expected number " << expectedNumVertices;
        errMessage = ss.str();
        return false;
    }

    return true;
}

bool SolutionInspector::fragmentSignatureIdsMatchPositionInVector(std::string& errMessage) {
    for (int i = 0; i < m_fragmentSignatures.size(); i++) {
        if (m_fragmentSignatures[i].getId() != i) {
            std::stringstream ss;
            ss << "FragmentSignature at position " << i << " should have matching id " << i << ", instead it has id " << m_fragmentSignatures[i].getId();
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

bool SolutionInspector::allFragmentSignaturesInitialized(std::string& errMessage) {
    for (int i = 0; i < m_fragmentSignatures.size(); i++) {
        FragmentSignature sig = m_fragmentSignatures[i];
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
    for (int i = 0; i < m_signatureIds.size(); i++) {
        int signatureId = m_signatureIds[i];
        if (signatureId < 0 || signatureId > m_fragmentSignatures.size()) {
            std::stringstream ss;
            ss << "Vertex " << i << " has invalid signature Id " << signatureId;
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

