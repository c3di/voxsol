#include <stdafx.h>
#include "Solution.h"
#include "problem/ProblemFragment.h"

Solution::Solution(DiscreteProblem& problem) :
    size(problem.getSize() + ettention::Vec3ui(1,1,1)),
    voxelSize(problem.getVoxelSize()),
    problem(&problem),
    signatureIds(size.x * size.y * size.z, -1),
    displacements(3 * size.x * size.y * size.z, 0)
{

}

Solution::~Solution() {

}

const std::vector<unsigned short>* Solution::getSignatureIds() const {
    return &signatureIds;
}

const std::vector<FragmentSignature>* Solution::getFragmentSignatures() const {
    return &fragmentSignatures;
}

std::vector<REAL>* Solution::getDisplacements() {
    return &displacements;
}

unsigned int Solution::mapToIndex(ettention::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        throw std::invalid_argument("given coordinate cannot be mapped to an index because it is outside the solution space");
    }
    return coordinate.x + coordinate.y * size.x + coordinate.z * size.x * size.y;
}

ettention::Vec3ui Solution::mapToCoordinate(unsigned int index) const {
    return ettention::Vec3ui(index % size.x, (index / size.x) % size.y, index / (size.x * size.y));
}

bool Solution::outOfBounds(ettention::Vec3ui& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= size.x || coordinate.y < 0 || coordinate.y >= size.y || coordinate.z < 0 || coordinate.z >= size.z;
}

void Solution::precomputeMatrices() {
    // This is separated into two steps to allow matrix pre-computation to be done asynchronously later
    gatherUniqueFragmentSignatures();
    precomputeMatricesForSignatures();
}

void Solution::gatherUniqueFragmentSignatures() {
    unsigned short signatureIdCounter = 0;
    std::unordered_map < ProblemFragmentKey, unsigned short> signatureToId;

    for (unsigned int z = 0; z < size.z; z++) {
        for (unsigned int y = 0; y < size.y; y++) {
            for (unsigned int x = 0; x < size.x; x++) {
                ettention::Vec3ui centerCoord(x, y, z);
                ProblemFragment fragment = problem->extractLocalProblem(centerCoord);
                ProblemFragmentKey materialConfiguration = fragment.getKey();

                if (signatureToId.count(materialConfiguration) <= 0) {
                    signatureToId[materialConfiguration] = signatureIdCounter;
                    signatureIdCounter++;
                }
                signatureIds[mapToIndex(centerCoord)] = signatureToId[materialConfiguration];
            }
        }
    }
    fragmentSignatures.resize(signatureIdCounter);
}

void Solution::precomputeMatricesForSignatures() {
    MatrixPrecomputer precomputer(voxelSize);

    for (int i = 0; i < signatureIds.size(); i++) {
        int signatureId = signatureIds[i];
        FragmentSignature* signature = &fragmentSignatures[signatureId];

        if (signature->getId() == USHRT_MAX) {
            // This matrix store hasn't been initialized yet so lets pre-compute the matrices
            signature->setId(signatureId);
            ettention::Vec3ui centerCoord = mapToCoordinate(i);
            ProblemFragment fragment = problem->extractLocalProblem(centerCoord);
            ProblemFragmentKey materialConfig = fragment.getKey();
            precomputer.initializeSignatureForFragment(signature, fragment);
        }
    }
}
