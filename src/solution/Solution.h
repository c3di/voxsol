#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <libmmv/math/Vec3.h>
#include "solution/FragmentSignature.h"
#include "problem/DiscreteProblem.h"
#include "solution/MatrixPrecomputer.h"

class Solution {
public:

    Solution(DiscreteProblem& problem);
    ~Solution();

    void precomputeMatrices();

    unsigned int mapToIndex(ettention::Vec3ui& coordinate) const;
    ettention::Vec3ui mapToCoordinate(unsigned int index) const;

    const std::vector<unsigned short>* getSignatureIds() const;
    const std::vector<FragmentSignature>* getFragmentSignatures() const;
    std::vector<REAL>* getDisplacements();
    unsigned short getSignatureIdForKey(const ProblemFragmentKey& key) const;

protected:
    const ettention::Vec3ui m_size;
    const ettention::Vec3<REAL> m_voxelSize;
    const DiscreteProblem* const m_problem;
    std::vector<unsigned short> m_signatureIds;
    std::vector<REAL> m_displacements;
    std::vector<FragmentSignature> m_fragmentSignatures;
    std::unordered_map < ProblemFragmentKey, unsigned short> m_signatureToId;

    void gatherUniqueFragmentSignatures();
    void precomputeMatricesForSignatures();

    inline bool outOfBounds(ettention::Vec3ui& coordinate) const;
};
