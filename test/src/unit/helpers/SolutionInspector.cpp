#include "helpers/SolutionInspector.h"
#include <sstream>

SolutionInspector::SolutionInspector(DiscreteProblem& problem) :
    Solution(problem)
{

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

bool SolutionInspector::matrixStoreIdsMatchPositionInVector(std::string& errMessage) {
    for (int i = 0; i < m_matrixStore.size(); i++) {
        if (m_matrixStore[i].getId() != i) {
            std::stringstream ss;
            ss << "MatrixStore at position " << i << " should have matching id " << i << ", instead it has id " << m_matrixStore[i].getId();
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

bool SolutionInspector::allMatrixStoresInitialized(std::string& errMessage) {
    for (int i = 0; i < m_matrixStore.size(); i++) {
        MatrixStore mStore = m_matrixStore[i];
        if (mStore.getId() < 0) {
            std::stringstream ss;
            ss << "MatrixStore " << i << " has invalid id " << mStore.getId();
            errMessage = ss.str();
            return false;
        }
        const Matrix3x3* lhs = mStore.getLHS();
        if (lhs == NULL || *lhs == Matrix3x3::identity) {
            std::stringstream ss;
            ss << "MatrixStore " << i << " has invalid LHS matrix";
            errMessage = ss.str();
            return false;
        }
        for (int j = 0; j < 27; j++) {
            const Matrix3x3* rhs = mStore.getRHS(j);
            if (j == 13) {
                if (rhs == NULL || *rhs != Matrix3x3::identity) {
                    std::stringstream ss;
                    ss << "MatrixStore " << i << " has invalid RHS matrix for center vertex 13";
                    errMessage = ss.str();
                    return false;
                }
            } else if (rhs == NULL || *rhs == Matrix3x3::identity) {
                std::stringstream ss;
                ss << "MatrixStore " << i << " has invalid RHS matrix for vertex " << j;
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
        if (signatureId < 0 || signatureId > m_matrixStore.size()) {
            std::stringstream ss;
            ss << "Vertex " << i << " has invalid signature Id " << signatureId;
            errMessage = ss.str();
            return false;
        }
    }
    return true;
}

