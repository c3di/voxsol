#include "stdafx.h"
#include "material/Material.h"


Material Material::EMPTY = Material(0, 0, 0);

Material::Material() :
    id(255),
    mu(0),
    lambda(0),
    poissonRatio(0),
    youngsModulus(0)
{

}

Material::Material(REAL eModul, REAL poissonRatio, unsigned char id) :
    lambda(calculateLambda(eModul, poissonRatio)),
    mu(calculateMu(eModul, poissonRatio)),
    id(id),
    poissonRatio(poissonRatio),
    youngsModulus(eModul)
{
    
}

Material::Material(const Material& other) : 
    lambda(other.lambda),
    mu(other.mu),
    id(other.id),
    poissonRatio(other.poissonRatio),
    youngsModulus(other.youngsModulus)
{

}

Material::~Material() {

}

REAL Material::calculateMu(REAL eModul, REAL poissonRatio) {
    return asREAL(eModul / (2.0 * (1.0 + poissonRatio)));
}

REAL Material::calculateLambda(REAL eModul, REAL poissonRatio) {
    REAL divisor = asREAL((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
    return (eModul * poissonRatio) / divisor;
}

bool Material::operator==(const Material& other) const
{
    return other.id == id;
}
