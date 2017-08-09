#include <stdafx.h>
#include "MaterialConfigurationEquations.h"

const size_t MaterialConfigurationEquations::SizeInBytes = sizeof(REAL) * 9 * 27;

MaterialConfigurationEquations::MaterialConfigurationEquations(unsigned short id) : 
    id(id)
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

MaterialConfigurationEquations::MaterialConfigurationEquations() :
    id(USHRT_MAX)
{
    matrices = std::vector<Matrix3x3>(27, Matrix3x3::identity);
}

MaterialConfigurationEquations::~MaterialConfigurationEquations() {

}

unsigned short MaterialConfigurationEquations::getId() {
    return id;
}

Matrix3x3* MaterialConfigurationEquations::getMatrices() {
    return matrices.data();
}

const Matrix3x3* MaterialConfigurationEquations::getLHS() const {
    return &matrices[13];
}

const Matrix3x3* MaterialConfigurationEquations::getRHS(unsigned int nodeIndex) const {
    return &matrices[nodeIndex];
}

bool MaterialConfigurationEquations::isInitialized() {
    return id != USHRT_MAX;
}

void MaterialConfigurationEquations::serialize(void* destination) const {
    Matrix3x3* dest = (Matrix3x3*)destination;

    for (unsigned int i = 0; i < 27; i++) {
        matrices[i].serialize(&dest[i]);
    }
}

void MaterialConfigurationEquations::setId(unsigned short id) {
    this->id = id;
}

void MaterialConfigurationEquations::setLHS(Matrix3x3& lhs) {
    matrices[13] = Matrix3x3(lhs);
}

void MaterialConfigurationEquations::setRHS(unsigned int nodeIndex, Matrix3x3& rhs) {
    matrices[nodeIndex] = Matrix3x3(rhs);
}
