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

    const std::vector<int>& getSignatureIds() const;
    const std::vector<MatrixStore>& getMatrixStore() const;
    int getSignatureIdForKey(const ProblemFragmentKey& key) const;

protected:
    const ettention::Vec3ui m_size;
    const ettention::Vec3<REAL> m_voxelSize;
    const DiscreteProblem* const m_problem;
    std::vector<int> m_signatureIds;
    std::vector<MatrixStore> m_matrixStore;
    std::unordered_map < ProblemFragmentKey, int> m_signatureToId;

    void gatherUniqueFragmentSignatures();
    void precomputeMatricesForSignatures();

    inline bool outOfBounds(ettention::Vec3ui& coordinate) const;
};
