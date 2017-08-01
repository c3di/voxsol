#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <libmmv/math/Vec3.h>
#include "solution/MatrixStore.h"
#include "problem/DiscreteProblem.h"
#include "solution/MatrixPrecomputer.h"

class Solution {
public:

    Solution(DiscreteProblem& problem);
    ~Solution();

    void precomputeMatrices();

    unsigned int mapToIndex(ettention::Vec3ui& coordinate) const;
    ettention::Vec3ui mapToCoordinate(unsigned int index) const;

    const std::vector<unsigned int>& getFragmentIds();
    const std::vector<MatrixStore>& getMatrixStore();
    unsigned int getFragmentIdForKey(const ProblemFragmentKey& key);

private:
    const ettention::Vec3ui m_size;
    const ettention::Vec3<REAL> m_voxelSize;
    const DiscreteProblem* const m_problem;
    std::vector<unsigned int> m_fragmentIds;
    std::vector<MatrixStore> m_matrixStore;
    std::unordered_map < ProblemFragmentKey, unsigned int> m_hashmap;

    void processNode(ettention::Vec3ui centerCoord, const MatrixPrecomputer* precomputer);
    inline bool outOfBounds(ettention::Vec3ui& coordinate) const;

};
