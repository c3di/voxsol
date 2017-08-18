#include <stdafx.h>
#include "MaterialConfigurationEquations.h"

// 27 3x3 matrices
const size_t MaterialConfigurationEquations::SizeInBytes = sizeof(REAL) * 9 * 27;

MaterialConfigurationEquations::MaterialConfigurationEquations(ConfigId id) :
    id(id)
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

MaterialConfigurationEquations::MaterialConfigurationEquations() :
    id(std::numeric_limits<ConfigId>::max())
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

MaterialConfigurationEquations::MaterialConfigurationEquations(ConfigId id, DirichletBoundary& condition) :
    id(id),
    dirichletBoundaryCondition(condition)
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
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

bool MaterialConfigurationEquations::isInitialized() {
    return id != std::numeric_limits<ConfigId>::max();
}

void MaterialConfigurationEquations::serialize(void* destination) const {
    unsigned char* serializationPointer = (unsigned char*)destination;

    for (unsigned char i = 0; i < 27; i++) {
        matrices[i].serialize(serializationPointer);
        serializationPointer += Matrix3x3::SizeInBytes;
    }

}

void MaterialConfigurationEquations::setId(ConfigId id) {
    this->id = id;
}

void MaterialConfigurationEquations::setLHSInverse(Matrix3x3& lhsInverse) {
    matrices[13] = Matrix3x3(lhsInverse);
}

void MaterialConfigurationEquations::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    matrices[nodeIndex] = Matrix3x3(rhs);
}
