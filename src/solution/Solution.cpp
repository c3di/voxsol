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

unsigned int Solution::mapToIndex(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        throw std::invalid_argument("given coordinate cannot be mapped to an index because it is outside the solution space");
    }
    return coordinate.x + coordinate.y * m_size.x + coordinate.z * m_size.x * m_size.y;
}

ettention::Vec3ui Solution::mapToCoordinate(unsigned int index) const {
    return ettention::Vec3ui(index % m_size.x, (index / m_size.x) % m_size.y, index / (m_size.x * m_size.y));
}

bool Solution::outOfBounds(ettention::Vec3ui& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= m_size.x || coordinate.y < 0 || coordinate.y >= m_size.y || coordinate.z < 0 || coordinate.z >= m_size.z;
}

void Solution::precomputeMatrices() {
    // This is separated into two steps to allow matrix pre-computation to be done asynchronously later
    gatherUniqueFragmentSignatures();
    precomputeMatricesForSignatures();
}

void Solution::gatherUniqueFragmentSignatures() {
    unsigned short signatureIdCounter = 0;
    std::unordered_map < ProblemFragmentKey, unsigned short> signatureToId;

    for (unsigned int z = 0; z < m_size.z; z++) {
        for (unsigned int y = 0; y < m_size.y; y++) {
            for (unsigned int x = 0; x < m_size.x; x++) {
                ettention::Vec3ui centerCoord(x, y, z);
                ProblemFragment fragment = m_problem->extractLocalProblem(centerCoord);
                ProblemFragmentKey materialConfiguration = fragment.key();

                if (signatureToId.count(materialConfiguration) <= 0) {
                    signatureToId[materialConfiguration] = signatureIdCounter;
                    signatureIdCounter++;
                }
                m_signatureIds[mapToIndex(centerCoord)] = signatureToId[materialConfiguration];
            }
        }
    }
    m_fragmentSignatures.resize(signatureIdCounter);
}

void Solution::precomputeMatricesForSignatures() {
    MatrixPrecomputer precomputer(m_voxelSize);

    for (int i = 0; i < m_signatureIds.size(); i++) {
        int signatureId = m_signatureIds[i];
        FragmentSignature* signature = &m_fragmentSignatures[signatureId];

        if (signature->getId() == USHRT_MAX) {
            // This matrix store hasn't been initialized yet so lets pre-compute the matrices
            signature->setId(signatureId);
            ettention::Vec3ui centerCoord = mapToCoordinate(i);
            ProblemFragment fragment = m_problem->extractLocalProblem(centerCoord);
            ProblemFragmentKey materialConfig = fragment.key();
            precomputer.initializeSignatureForFragment(signature, fragment);
        }
    }
}
