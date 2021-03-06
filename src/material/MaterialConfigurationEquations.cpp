#include <stdafx.h>
#include "MaterialConfigurationEquations.h"

// 28 3x3 matrices, 1 vec3 for neumann boundary condition (stress)
const size_t MaterialConfigurationEquations::SizeInBytes = sizeof(REAL) * 9 * 28 + 3 * sizeof(REAL);

MaterialConfigurationEquations::MaterialConfigurationEquations(ConfigId id) :
    id(id),
    neumannBoundaryCondition()
{
    matrices = std::vector<Matrix3x3>(28, Matrix3x3::identity);
    dirichlet[0] = 1; dirichlet[1] = 1; dirichlet[2] = 1;
}

MaterialConfigurationEquations::MaterialConfigurationEquations() :
    id(std::numeric_limits<ConfigId>::max()),
    neumannBoundaryCondition()
{
    matrices = std::vector<Matrix3x3>(28, Matrix3x3::identity);
}

MaterialConfigurationEquations::~MaterialConfigurationEquations() {

}

ConfigId MaterialConfigurationEquations::getId() {
    return id;
}

Matrix3x3* MaterialConfigurationEquations::getMatrices() {
    return matrices.data();
}

const Matrix3x3* MaterialConfigurationEquations::getLHSInverse() const {
    return &matrices[13];
}

const Matrix3x3* MaterialConfigurationEquations::getRHS(unsigned int nodeIndex) const {
    return &matrices[nodeIndex];
}

const NeumannBoundary* MaterialConfigurationEquations::getNeumannBoundaryCondition() const {
    return &neumannBoundaryCondition;
}

bool MaterialConfigurationEquations::isInitialized() {
    return id != std::numeric_limits<ConfigId>::max();
}

//#define optimized_mem

void MaterialConfigurationEquations::serialize(void* destination) const {
#ifdef optimized_mem
    REAL* serializationPointer = (REAL*)destination;

    // Matrices are serialized for optimal memory access in CUDA:
    // Neighbor 0 row 0 col 0, Neighbor 1 row 0 col 0... Neighbor 26 row 0 col 0, Neighbor 0 row 0 col 1, Neighbor 1 row 0 col 1 ... Neighbor 26 row 0 col 1, Neighbor 0 row 0 col 2...
    // See buildRHSVectorForVertex in SolveDisplacementCuda.cu

    for (unsigned char row = 0; row < 3; row++) {
        for (unsigned char col = 0; col < 3; col++) {

            for (unsigned char neighbor = 0; neighbor < 27; neighbor++) {
                *serializationPointer = matrices[neighbor].at(col, row);
                serializationPointer += 1;
            }
        }
    }

    memcpy(serializationPointer, &neumannBoundaryCondition.stress.dim, sizeof(REAL)*3);

#else
    unsigned char* serializationPointer = (unsigned char*)destination;
    for (unsigned char i = 0; i < 28; i++) {
        matrices[i].serialize(serializationPointer);
        serializationPointer += Matrix3x3::SizeInBytes;
    }
    memcpy(serializationPointer, &neumannBoundaryCondition.force.dim, sizeof(REAL) * 3);

#endif
}

void MaterialConfigurationEquations::setId(ConfigId id) {
    this->id = id;
}

void MaterialConfigurationEquations::setLHS(Matrix3x3& lhs) {
    matrices[27] = Matrix3x3(lhs);
}

void MaterialConfigurationEquations::setLHSInverse(Matrix3x3& lhsInverse) {
    matrices[13] = Matrix3x3(lhsInverse);
}

void MaterialConfigurationEquations::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    matrices[nodeIndex] = Matrix3x3(rhs);
}

void MaterialConfigurationEquations::setNeumannBoundaryCondition(const NeumannBoundary& stress) {
    neumannBoundaryCondition = stress;
}
