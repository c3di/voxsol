#include <stdafx.h>
#include "Solution.h"
#include "problem/ProblemFragment.h"

Solution::Solution(DiscreteProblem& problem) :
    m_size(problem.getSize() + ettention::Vec3ui(1,1,1)),
    m_voxelSize(problem.getVoxelSize()),
    m_problem(&problem),
    m_signatureIds(m_size.x * m_size.y * m_size.z, -1),
    m_displacements(3 * m_size.x * m_size.y * m_size.z, 0)
{

}

Solution::~Solution() {

}

const std::vector<unsigned short>* Solution::getSignatureIds() const {
    return &m_signatureIds;
}

const std::vector<FragmentSignature>* Solution::getFragmentSignatures() const {
    return &m_fragmentSignatures;
}

std::vector<REAL>* Solution::getDisplacements() {
    return &m_displacements;
}

unsigned short Solution::getSignatureIdForKey(const ProblemFragmentKey& key) const {
    if (m_signatureToId.count(key) <= 0) {
        throw std::exception("the given problem fragment key is not known in this solution");
    }
    return m_signatureToId.at(key);
}

unsigned int Solution::mapToIndex(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        throw std::invalid_argument("given coordinate cannot be mapped to an index because it is outside the solution space");
    }
    return coordinate.x + coordinate.y * m_size.x + coordinate.z * m_size.x * m_size.y;
}

ettention::Vec3ui Solution::mapToCoordinate(unsigned int index) const {
    return ettention::Vec3ui(index % m_size.x, (index / m_size.x) % m_size.y, index / (m_size.x * m_size.y));
}

inline bool Solution::outOfBounds(ettention::Vec3ui& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= m_size.x || coordinate.y < 0 || coordinate.y >= m_size.y || coordinate.z < 0 || coordinate.z >= m_size.z;
}

void Solution::precomputeMatrices() {
    // This is separated into two steps to allow matrix pre-computation to be done asynchronously later
    gatherUniqueFragmentSignatures();
    precomputeMatricesForSignatures();
}

void Solution::gatherUniqueFragmentSignatures() {
    unsigned short signatureIdCounter = 0;

    for (unsigned int z = 0; z < m_size.z; z++) {
        for (unsigned int y = 0; y < m_size.y; y++) {
            for (unsigned int x = 0; x < m_size.x; x++) {
                ettention::Vec3ui centerCoord(x, y, z);
                ProblemFragment fragment = m_problem->extractLocalProblem(centerCoord);
                ProblemFragmentKey materialConfig = fragment.key();

                if (m_signatureToId.count(materialConfig) <= 0) {
                    m_signatureToId[materialConfig] = signatureIdCounter;
                    signatureIdCounter++;
                }
                m_signatureIds[mapToIndex(centerCoord)] = m_signatureToId[materialConfig];
            }
        }
    }
    m_fragmentSignatures.resize(signatureIdCounter);
}

void Solution::precomputeMatricesForSignatures() {
    MatrixPrecomputer precomputer(m_voxelSize);

    for (int i = 0; i < m_signatureIds.size(); i++) {
        int signatureId = m_signatureIds[i];
        FragmentSignature* store = &m_fragmentSignatures[signatureId];

        if (store->getId() == USHRT_MAX) {
            // This matrix store hasn't been initialized yet so lets pre-compute the matrices
            store->setId(signatureId);
            ettention::Vec3ui centerCoord = mapToCoordinate(i);
            ProblemFragment fragment = m_problem->extractLocalProblem(centerCoord);
            ProblemFragmentKey materialConfig = fragment.key();
            precomputer.initializeSignatureForFragment(store, fragment);
        }
    }
}
