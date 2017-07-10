#include <stdafx.h>
#include "Solution.h"
#include "problem/ProblemFragment.h"

Solution::Solution(DiscreteProblem& problem) :
    m_size(problem.getSize() + ettention::Vec3ui(1,1,1)),
    m_voxelSize(problem.getVoxelSize()),
    m_problem(&problem),
    m_nodes(m_size.x * m_size.y * m_size.z, 0)
{

}

Solution::~Solution() {

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
    MatrixPrecomputer precomputer(m_voxelSize);

    for (unsigned int z = 0; z < m_size.z; z++) {
        for (unsigned int y = 0; y < m_size.y; y++) {
            for (unsigned int x = 0; x < m_size.x; x++) {
                ettention::Vec3ui centerCoord(x, y, z);
                processNode(centerCoord, &precomputer);
            }
        }
    }
}

void Solution::processNode(ettention::Vec3ui centerCoord, const MatrixPrecomputer* precomputer) {
    ProblemFragment fragment = m_problem->extractLocalProblem(centerCoord);
    ProblemFragmentKey materialConfig = fragment.key();
    unsigned int idForNode = 0;
    
    //TODO: Generate IDs for all unique configs first, then compute their matrices in a second (possibly async) step
    if (m_hashmap.count(materialConfig) > 0) {
        idForNode = m_hashmap.at(materialConfig);
    }
    else {
        MatrixStore matStore = precomputer->computeMatrixStoreForFragment(fragment);
        m_matrixStore.push_back(matStore);
        idForNode = (unsigned int) m_matrixStore.size() - 1;
        m_hashmap[materialConfig] = idForNode;
    }

    unsigned int index = mapToIndex(centerCoord);
    m_nodes[index] = idForNode;
}
